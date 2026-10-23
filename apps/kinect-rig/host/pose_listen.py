#!/usr/bin/env python3
"""Record the rig phone's ARCore pose stream and report tracking quality and clock offset.

usage: pose_listen.py [--seconds 30] [--adb PATH] [--serial SERIAL] [-o poses.jsonl]

Forwards tcp:8765 through adb (works over USB or wireless debugging), pings the phone twice
a second to estimate the phone↔laptop clock offset (lowest round trip wins), and writes every
line it receives, plus the offset, to a JSONL file.
"""
import argparse
import json
import math
import os
import socket
import subprocess
import time

PORT = 8765


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=float, default=30)
    ap.add_argument('--adb', default=os.path.expanduser('~/scanrig/platform-tools/adb'))
    ap.add_argument('--serial', default=os.environ.get('ANDROID_SERIAL'))
    ap.add_argument('-o', default='poses.jsonl')
    a = ap.parse_args()

    adb = [a.adb] + (['-s', a.serial] if a.serial else [])
    subprocess.run(adb + ['forward', f'tcp:{PORT}', f'tcp:{PORT}'], check=True)
    sock = socket.create_connection(('127.0.0.1', PORT), timeout=5)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(0.1)

    frames, pongs, sent = [], [], {}
    buf = b''
    start = time.monotonic()
    next_ping, ping_id = start, 0
    with open(a.o, 'w') as out:
        while time.monotonic() - start < a.seconds:
            now = time.monotonic()
            if now >= next_ping:
                ping_id += 1
                sent[ping_id] = time.monotonic_ns()
                sock.sendall(f'ping {ping_id}\n'.encode())
                next_ping = now + 0.5
            try:
                chunk = sock.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                break
            buf += chunk
            *lines, buf = buf.split(b'\n')
            arrived = time.monotonic_ns()
            for raw in lines:
                msg = json.loads(raw)
                msg['laptop_ns'] = arrived
                if msg['type'] == 'pong' and msg['id'] in sent:
                    t0 = sent.pop(msg['id'])
                    msg['rtt_ns'] = arrived - t0
                    msg['offset_ns'] = msg['phone_ns'] - (t0 + arrived) // 2  # phone clock - laptop clock
                    pongs.append(msg)
                elif msg['type'] == 'frame':
                    frames.append(msg)
                out.write(json.dumps(msg) + '\n')
        best = min(pongs, key=lambda p: p['rtt_ns']) if pongs else None
        if best:
            out.write(json.dumps({'type': 'clock', 'offset_ns': best['offset_ns'], 'rtt_ns': best['rtt_ns']}) + '\n')
    sock.close()
    report(frames, pongs, best, a.o)


def report(frames, pongs, best, path):
    if not frames:
        print('no frames received: is the app open on the phone and showing "waiting on :8765"?')
        return
    span = (frames[-1]['t_ns'] - frames[0]['t_ns']) / 1e9
    tracked = [f for f in frames if f['tracking'] == 'TRACKING']
    reasons = {}
    for f in frames:
        if f['tracking'] != 'TRACKING':
            reasons[f['reason']] = reasons.get(f['reason'], 0) + 1
    path_len = sum(math.dist(p['pose'][:3], q['pose'][:3]) for p, q in zip(tracked, tracked[1:]))
    pos = [f['pose'][:3] for f in tracked]
    extent = [max(c) - min(c) for c in zip(*pos)] if pos else [0, 0, 0]
    gaps = [(q['t_ns'] - p['t_ns']) / 1e6 for p, q in zip(frames, frames[1:])]
    k = next((f['intrinsics'] for f in frames if 'intrinsics' in f), None)
    print(f'{len(frames)} frames over {span:.1f} s ({len(frames) / max(span, 1e-9):.1f} fps), '
          f'max gap {max(gaps, default=0):.0f} ms')
    print(f'tracking {len(tracked) / len(frames):.0%}' + (f', not tracking: {reasons}' if reasons else ''))
    print(f'path {path_len:.2f} m, extent x/y/z {extent[0]:.2f} / {extent[1]:.2f} / {extent[2]:.2f} m')
    if k:
        print(f"CPU image {k['width']}x{k['height']}, fx {k['fx']:.1f} fy {k['fy']:.1f}, "
              f"hfov {math.degrees(2 * math.atan(k['width'] / 2 / k['fx'])):.1f} deg")
    if best:
        rtts = sorted(p['rtt_ns'] / 1e6 for p in pongs)
        print(f'clock: {len(pongs)} pings, rtt min {rtts[0]:.1f} / median {rtts[len(rtts) // 2]:.1f} ms; '
              f'offset ±{best["rtt_ns"] / 2e6:.1f} ms')
    print(f'→ {path}')


if __name__ == '__main__':
    main()
