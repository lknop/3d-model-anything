# Kinect rig: room capture without a LiDAR iPhone

An alternative to the [iPhone app](../house-capture/README.md). A **Kinect for Windows v1** supplies
depth, and an **Android phone** rigidly attached to it supplies ARCore tracking and photos. A Linux
laptop records both. The tools turn a recording into a `house-capture/5` session bundle, so the
[scan-to-threejs skill](../../.agents/skills/scan-to-threejs/SKILL.md) and its scripts work on it.

Status: working end to end on one cable-tethered trial scan of an attic. Calibration, tracking
correction and layout detection are new and lightly tested; see [Known limits](#known-limits).

## Hardware

- Kinect for Windows v1 (Xbox 360 Kinect should work too). It needs 12 V (~1 A) from its adapter;
  USB alone cannot power it. For walking around, use a power bank with an AC outlet, or a 12 V
  USB-C PD trigger cable with a third-party Kinect adapter that has a barrel input.
- An ARCore phone, attached **rigidly to the Kinect's sensor bar**, not its tilting base. Portrait or
  landscape both work. Tested: ASUS Zenfone 8 (1920x1080 CPU images at 30 fps).
- A Linux laptop with an NVIDIA GPU (tested: RTX 3050 Ti, Ubuntu 22.04-based TUXEDO OS, driver 560).
- A printed checkerboard: 10 x 7 squares (9 x 6 inner corners), 25.4 mm, at 100% scale, lying flat.

## Setup (laptop)

Everything except the udev rule lives in user space, here under `~/scanrig`:

```sh
# libfreenect 0.7.5 (Ubuntu packages only 0.5.3); needs libusb-1.0-0-dev
git clone --branch v0.7.5 https://github.com/OpenKinect/libfreenect.git
cmake -S libfreenect -B build -DCMAKE_INSTALL_PREFIX=$HOME/scanrig/local -DBUILD_EXAMPLES=OFF \
      -DBUILD_C_SYNC=ON -DBUILD_PYTHON=OFF -DBUILD_PYTHON3=OFF -DBUILD_REDIST_PACKAGE=ON
cmake --build build -j8 && cmake --install build
# udev: allow the plugdev group to open the Kinect (045e:02b0/02ad/02ae/02c2/02be/02bf), MODE 0660

uv venv -p 3.11 ~/scanrig/.venv
uv pip install -p ~/scanrig/.venv/bin/python open3d numpy pillow opencv-python-headless scipy
# Google's platform-tools for adb (wireless debugging needs adb >= 30)
# JDK 17, Android command-line tools (platform 35, build-tools 35) and Gradle 8.10 to build the app
```

Closing the lid suspends the laptop by default, which ends a recording. `rig_record.py` holds a
systemd inhibitor while recording, but logind grants that only to the active local session. For
recordings started over SSH, ignore the lid while on external power (a power bank counts):

```sh
sudo mkdir -p /etc/systemd/logind.conf.d
printf '[Login]\nHandleLidSwitchExternalPower=ignore\n' | sudo tee /etc/systemd/logind.conf.d/scanrig.conf
sudo reboot
```

Build and install the phone app from `android/`: write `local.properties` with
`sdk.dir=<android-sdk>`, run `JAVA_HOME=<jdk17> gradle assembleDebug`, then
`adb install -r app/build/outputs/apk/debug/app-debug.apk`. The phone can use USB or
**wireless debugging** (pair once with `adb pair`; later, `adb mdns services` shows the port).

## Workflow

`H` is this folder's `host/`, `S` a session folder. Each tool prints a summary; `--help` lists options.

```sh
# 1. Record (Ctrl-C or --seconds stops; the phone's own recording is pulled in at the end)
ANDROID_SERIAL=<phone> python $H/rig_record.py calib-001
#    Calibration: aim at the board from ~15 viewpoints, hold still until each beep,
#    then turn the rig around for 10-15 s. Re-calibrate whenever the phone moves on the rig.
python $H/rig_calibrate.py sessions/calib-001 -o rig.json --square 0.0254

# 2. A room: walk slowly, pause at "HOLD STILL" until the beep; point the rig up at the ceiling too
ANDROID_SERIAL=<phone> python $H/rig_record.py room-001

# 3. Kinect poses from ARCore x calibration, then fix ARCore's re-anchoring jumps with Kinect depth
python $H/rig_to_bundle.py $S --calibration rig.json --write-prior prior.txt
python $H/rig_refine.py $S --prior prior.txt --calibration rig.json -o refined.txt
python $H/rig_to_bundle.py $S --calibration rig.json --refined refined.txt -o bundle

# 4. The skill's own reconstruction, then the layout (stands in for RoomPlan)
python $SKILL/reconstruct_bundle.py bundle --max-depth 3.0 -o recon
python $H/rig_layout.py bundle --mesh recon/scan_world.ply --name "Attic" --plan layout.png
python $SKILL/bundle_to_room.py bundle --room 0 -o room.json --plan plan.png   # unchanged skill script

# Previews of any mesh (angled, top, height-coded); no display or GPU needed
python $H/render_mesh.py recon/scan_world.ply -o view
```

Kinect-only, no phone: `kinect_fusion.py <session-or-TUM-folder> -o out` tracks the Kinect alone (GPU),
then `rig_to_bundle.py $S --kinect-poses out/trajectory.txt -o bundle`. It has no gravity (the world is
the first camera) and drifts about 0.5% of the distance walked.

## Tools

| File | What it does |
|---|---|
| `android/` | Phone app. ARCore with a viewfinder; uses the largest 30 fps CPU image; streams poses over adb-forwarded TCP 8765; answers clock pings; records sessions on the phone (`poses.jsonl`, `keyframes.jsonl`, `images/`). A keyframe is due after 25 cm, 12 degrees or 3 s; it is taken on the first frame with predicted blur under 2.5 px, relaxing if the operator doesn't pause. |
| `host/kinect.py` | libfreenect through ctypes, **camera subdevice only** (see below). RGB plus depth registered to RGB, in mm. |
| `host/rig_record.py` | Session recorder: Kinect frames with laptop timestamps, phone start/stop, live pose stream, clock round trips, pull of the phone recording. |
| `host/rig_calibrate.py` | Kinect delay (image motion vs. ARCore angular speed), Kinect RGB intrinsics, and the phone-to-Kinect transform (joint `stereoCalibrate` with a bootstrap uncertainty). |
| `host/rig_to_bundle.py` | Session or TUM sequence → `house-capture/5` bundle (photos, Float16 depth frames, manifest). Photos record `upright` from gravity. |
| `host/rig_refine.py` | Splits the ARCore-derived trajectory at unphysical jumps; aligns each stretch's Kinect cloud to the earlier ones (point-to-plane ICP), one rigid correction per stretch. |
| `host/kinect_fusion.py` | Open3D GPU dense SLAM for Kinect-only tracking or fusing given poses; `--gt` reports ATE. |
| `host/rig_layout.py` | Floor, walls (Manhattan, outermost, reaching the floor), doors/openings (gaps the scan sees through), floor outline, object boxes. Ceiling as planar faces (`ceilingFaces`), their edges (`ceilingEdges`, peak/valley) and wall `topProfile`s. Writes `rooms/room-N.json` plus the manifest record. |
| `host/render_mesh.py` | OpenCV painter's-algorithm previews (Open3D's offscreen renderer returns blank frames above ~65k vertices). |
| `host/test_rig_layout.py` | Synthetic hip-roof attic with a known answer. |

Formats beyond the iPhone contract: photo records may carry `upright` (`cw90`, the iPhone default,
`ccw90`, `180` or `none`), which `bundle_to_room.py` and `prepare_photos.py` honor. Room JSON adds
`ceilingFaces`, `ceilingEdges`, `walls[].topProfile` and `ceilingHeight`. These are ignored by
RoomPlan-only readers.

## Measured so far

- **Kinect-only tracking** (Open3D, depth-only ICP, no loop closure) on TUM RGB-D: fr1/desk 6.2 cm,
  fr3/long_office 10.0 cm ATE RMSE at ~40 fps. That is 2-3x Kintinuous and 3-6x ElasticFusion, so the
  rig relies on ARCore for tracking.
- **ARCore** (Zenfone 8): 100% tracking on room walks. It re-anchors its map several times per
  scan, with jumps of 5-17 cm and **7-9 degrees of heading**; `rig_refine.py` corrects them to a
  ~1.5 cm residual. Floor thickness on the trial went from 40 mm to 17 mm.
- **Calibration** (taped rig): +35 ms Kinect delay, phone-to-Kinect uncertainty 1-5 mm and 0.06 degrees.
- **Clock sync** over wireless adb: +/-3-4 ms. ARCore camera timestamps use the elapsedRealtime base.

## Known limits

- Kinect for Windows: opening the motor or LED resets its audio device and drops the session, so
  only the camera is opened. The accelerometer needs Microsoft firmware and is unused; gravity
  comes from ARCore.
- Kinect v1 depth is coarse beyond ~3 m and fails in sunlight, on glass and on black surfaces.
- Layout: walls are assumed to meet at right angles; windows are not detected; furniture is a set of
  unlabeled boxes; ceilings are only as complete as the scan; stairs and multi-room joins are not handled.
- Indoor exposures are 20-30 ms: photos are sharp only if the operator pauses at the beep.
