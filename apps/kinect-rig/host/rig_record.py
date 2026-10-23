#!/usr/bin/env python3
"""Record one rig session: Kinect RGB + registered depth on the laptop, ARCore poses + keyframes on the phone.

usage: rig_record.py [name] [--seconds N] [--no-phone] [--no-kinect] [--root DIR] [--serial SERIAL]

Stops after --seconds, or on Ctrl-C. Writes <root>/<name>/:
  session.json             devices, counts, clock offset
  kinect/frames.jsonl      one line per frame: stream (rgb|depth), i, laptop_ns, device_ts, file
  kinect/rgb/NNNNNN.jpg    640x480 RGB
  kinect/depth/NNNNNN.png  16-bit depth in millimeters, registered to the RGB image (0 = no reading)
  phone/stream.jsonl       the live pose stream as received, with laptop_ns added
  phone/clock.jsonl        ping round trips; offset_ns = phone clock - laptop clock
  phone/recording/         the phone's own files: poses.jsonl, keyframes.jsonl, images/, session.json
Laptop times are time.monotonic_ns() at arrival; phone times are SystemClock.elapsedRealtimeNanos.
"""
import argparse
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PORT = 8765
MAX_PENDING_WRITES = 90  # three seconds of both streams; beyond that frames are dropped and counted


def hold_awake():
    """Block lid-close suspend, sleep and idle for the recording. logind only grants this to the
    active local session; over SSH rely on HandleLidSwitchExternalPower=ignore (see README)."""
    try:
        lock = subprocess.Popen(['systemd-inhibit', '--what=handle-lid-switch:sleep:idle', '--who=rig_record',
                                 '--why=recording a scan', '--mode=block', 'sleep', 'infinity'],
                                stderr=subprocess.PIPE, text=True)
    except FileNotFoundError:
        return None
    time.sleep(0.3)
    if lock.poll() is not None:
        print(f'note: cannot hold the laptop awake ({lock.stderr.read().strip()}); closing the lid may suspend it')
        return None
    return lock


class PhoneLink:
    """The rig phone over `adb forward`: live poses, clock pings, start/stop commands."""

    def __init__(self, adb, out_dir):
        self.adb = adb
        subprocess.run(adb + ['forward', f'tcp:{PORT}', f'tcp:{PORT}'], check=True, capture_output=True)
        self.sock = socket.create_connection(('127.0.0.1', PORT), timeout=5)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(None)
        self.send_lock = threading.Lock()
        self.stream = open(os.path.join(out_dir, 'stream.jsonl'), 'w')
        self.clock = open(os.path.join(out_dir, 'clock.jsonl'), 'w')
        self.acks = queue.Queue()
        self.pings = {}
        self.best_clock = None
        self.frames = self.tracked = self.keyframes = 0
        self.last = None
        self.running = True
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.pinger = threading.Thread(target=self._ping, daemon=True)
        self.reader.start()
        self.pinger.start()

    def command(self, text):
        with self.send_lock:
            self.sock.sendall((text + '\n').encode())

    def wait_ack(self, cmd, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                ack = self.acks.get(timeout=0.2)
            except queue.Empty:
                continue
            if ack.get('cmd') == cmd:
                return ack
        return None

    def _ping(self):
        n = 0
        while self.running:
            n += 1
            self.pings[n] = time.monotonic_ns()
            try:
                self.command(f'ping {n}')
            except OSError:
                return
            time.sleep(0.5)

    def _read(self):
        buf = b''
        while self.running:
            try:
                chunk = self.sock.recv(65536)
            except OSError:
                break
            if not chunk:
                break
            arrived = time.monotonic_ns()
            buf += chunk
            *lines, buf = buf.split(b'\n')
            for raw in lines:
                msg = json.loads(raw)
                kind = msg.get('type')
                if kind == 'pong':
                    sent = self.pings.pop(msg['id'], None)
                    if sent is None:
                        continue
                    sample = {'id': msg['id'], 'laptop_sent_ns': sent, 'laptop_recv_ns': arrived, 'phone_ns': msg['phone_ns'],
                              'rtt_ns': arrived - sent, 'offset_ns': msg['phone_ns'] - (sent + arrived) // 2}
                    self.clock.write(json.dumps(sample) + '\n')
                    if self.best_clock is None or sample['rtt_ns'] < self.best_clock['rtt_ns']:
                        self.best_clock = sample
                elif kind == 'ack':
                    self.acks.put(msg)
                else:
                    msg['laptop_ns'] = arrived
                    self.stream.write(json.dumps(msg) + '\n')
                    if kind == 'frame':
                        self.frames += 1
                        self.tracked += msg['tracking'] == 'TRACKING'
                        self.last = msg
                    elif kind == 'keyframe':
                        self.keyframes += 1

    def pull(self, phone_dir, dest):
        tmp = dest + '.pulling'
        os.makedirs(tmp, exist_ok=True)
        result = subprocess.run(self.adb + ['pull', phone_dir, tmp], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        os.rename(os.path.join(tmp, os.path.basename(phone_dir.rstrip('/'))), dest)
        os.rmdir(tmp)

    def close(self):
        self.running = False
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()
        self.reader.join(timeout=2)
        self.stream.close()
        self.clock.close()


class KinectRecorder:
    """Kinect frames to disk on worker threads; frames.jsonl is written in arrival order."""

    def __init__(self, out_dir):
        import cv2
        from kinect import Kinect, WIDTH, HEIGHT
        self.cv2, self.w, self.h = cv2, WIDTH, HEIGHT
        self.dir = out_dir
        for sub in ('rgb', 'depth'):
            os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
        self.index = open(os.path.join(out_dir, 'frames.jsonl'), 'w')
        self.pool = ThreadPoolExecutor(4)
        self.pending = 0
        self.lock = threading.Lock()
        self.count = {'rgb': 0, 'depth': 0}
        self.dropped = {'rgb': 0, 'depth': 0}
        self.device = Kinect(lambda data, ts: self._frame('rgb', data, ts), lambda data, ts: self._frame('depth', data, ts))
        self.focal_estimate = self.device.focal_estimate()

    def _frame(self, stream, data, device_ts):
        laptop_ns = time.monotonic_ns()
        with self.lock:
            if self.pending >= MAX_PENDING_WRITES:
                self.dropped[stream] += 1
                return
            self.pending += 1
        i = self.count[stream]
        self.count[stream] += 1
        name = f'{stream}/{i:06d}.' + ('jpg' if stream == 'rgb' else 'png')
        self.index.write(json.dumps({'stream': stream, 'i': i, 'laptop_ns': laptop_ns, 'device_ts': device_ts, 'file': name}) + '\n')
        self.pool.submit(self._write, stream, data, os.path.join(self.dir, name))

    def _write(self, stream, data, path):
        try:
            if stream == 'rgb':
                rgb = np.frombuffer(data, np.uint8).reshape(self.h, self.w, 3)
                self.cv2.imwrite(path, rgb[:, :, ::-1], [self.cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                depth = np.frombuffer(data, np.uint16).reshape(self.h, self.w)
                self.cv2.imwrite(path, depth, [self.cv2.IMWRITE_PNG_COMPRESSION, 1])
        finally:
            with self.lock:
                self.pending -= 1

    def close(self):
        self.device.close()
        self.pool.shutdown(wait=True)
        self.index.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name', nargs='?', default=time.strftime('%Y%m%d-%H%M%S'))
    ap.add_argument('--seconds', type=float, default=0, help='stop after this long (default: Ctrl-C)')
    ap.add_argument('--no-phone', action='store_true')
    ap.add_argument('--no-kinect', action='store_true')
    ap.add_argument('--root', default=os.path.expanduser('~/scanrig/sessions'))
    ap.add_argument('--adb', default=os.path.expanduser('~/scanrig/platform-tools/adb'))
    ap.add_argument('--serial', default=os.environ.get('ANDROID_SERIAL'))
    a = ap.parse_args()

    out = os.path.join(a.root, a.name)
    if os.path.exists(out):
        ap.error(f'{out} exists; choose another name')
    os.makedirs(out)
    adb = [a.adb] + (['-s', a.serial] if a.serial else [])
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    phone = kinect = None
    awake = hold_awake()
    session = {'format': 'kinect-rig/1', 'name': a.name, 'laptopClock': 'time.monotonic_ns',
               'phoneClock': 'SystemClock.elapsedRealtimeNanos'}
    try:
        if not a.no_phone:
            os.makedirs(os.path.join(out, 'phone'))
            phone = PhoneLink(adb, os.path.join(out, 'phone'))
            phone.command(f'start {a.name}')
            ack = phone.wait_ack('start', 5)
            if not ack or not ack.get('ok'):
                raise RuntimeError(f'phone did not start recording: {ack}')
            session['phoneDir'] = ack['dir']
        if not a.no_kinect:
            kinect = KinectRecorder(os.path.join(out, 'kinect'))
            session['kinectFocalEstimate'] = kinect.focal_estimate
            kinect.device.start()
        session['startedNs'] = time.monotonic_ns()
        print(f'recording {out} (Ctrl-C to stop)')
        last_print, last_counts = time.monotonic(), (0, 0, 0)
        while not stop.is_set():
            if kinect:
                if not kinect.device.process_events():
                    print('\nKinect event loop failed')
                    break
            else:
                time.sleep(0.05)
            now = time.monotonic()
            if a.seconds and now - (session['startedNs'] / 1e9) >= a.seconds:
                break
            if now - last_print >= 1:
                counts = (kinect.count['rgb'] if kinect else 0, kinect.count['depth'] if kinect else 0, phone.frames if phone else 0)
                rates = [(c - p) / (now - last_print) for c, p in zip(counts, last_counts)]
                parts = [f'{now - session["startedNs"] / 1e9:6.1f}s']
                if kinect:
                    parts.append(f'kinect rgb {rates[0]:4.1f} depth {rates[1]:4.1f} fps, dropped {sum(kinect.dropped.values())}')
                if phone:
                    state = phone.last['tracking'] if phone.last else '-'
                    if phone.last and state != 'TRACKING':
                        state += f" ({phone.last['reason']})"
                    clock = f'±{phone.best_clock["rtt_ns"] / 2e6:.1f} ms' if phone.best_clock else '-'
                    parts.append(f'phone {state} {rates[2]:4.1f} fps, {phone.keyframes} keyframes, clock {clock}')
                print('\r' + ' | '.join(parts) + '   ', end='', flush=True)
                last_print, last_counts = now, counts
        print()
    finally:
        session['stoppedNs'] = time.monotonic_ns()
        if kinect:
            kinect.close()
            session['kinect'] = {'rgbFrames': kinect.count['rgb'], 'depthFrames': kinect.count['depth'], 'dropped': kinect.dropped,
                                 'width': kinect.w, 'height': kinect.h, 'depth': 'uint16 mm registered to rgb'}
        if phone:
            try:
                phone.command('stop')
                ack = phone.wait_ack('stop', 40)
                session['phone'] = ack.get('summary') if ack else None
                session['phoneStream'] = {'frames': phone.frames, 'tracked': phone.tracked, 'keyframes': phone.keyframes}
                session['clock'] = phone.best_clock
            finally:
                phone.close()
            if session.get('phoneDir'):
                print('pulling phone recording…')
                try:
                    phone.pull(session['phoneDir'], os.path.join(out, 'phone', 'recording'))
                except RuntimeError as e:
                    print(f'pull failed ({e}); retry with: adb pull {session["phoneDir"]} {out}/phone/')
        with open(os.path.join(out, 'session.json'), 'w') as f:
            json.dump(session, f, indent=2)
        if awake:
            awake.terminate()
    report(out, session)


def report(out, s):
    secs = (s['stoppedNs'] - s['startedNs']) / 1e9 if 'startedNs' in s else 0
    print(f'session {s["name"]}: {secs:.1f} s')
    if 'kinect' in s:
        k = s['kinect']
        print(f'  kinect: {k["rgbFrames"]} rgb / {k["depthFrames"]} depth frames ({k["depthFrames"] / max(secs, 1e-9):.1f} fps), '
              f'dropped {k["dropped"]}, focal estimate {s.get("kinectFocalEstimate") or 0:.1f} px')
    if 'phoneStream' in s:
        p, rec = s['phoneStream'], s.get('phone') or {}
        print(f'  phone: {p["frames"]} frames streamed, tracking {p["tracked"] / max(p["frames"], 1):.0%}, '
              f'{rec.get("keyframes", "?")} keyframes ({rec.get("keyframesWritten", "?")} written), camera {((rec.get("camera") or {}).get("chosen"))}')
    if s.get('clock'):
        print(f'  clock: phone - laptop = {s["clock"]["offset_ns"] / 1e6:.3f} ms ± {s["clock"]["rtt_ns"] / 2e6:.1f} ms')
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(out) for f in fs)
    print(f'  {size / 1e6:.0f} MB → {out}')


if __name__ == '__main__':
    main()
