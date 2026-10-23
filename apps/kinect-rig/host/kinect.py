"""Kinect v1 through libfreenect (ctypes): 640x480 RGB and RGB-registered depth in millimeters.

Opens the camera subdevice only. On Kinect for Windows (and the 1473 Xbox model) the motor and
LED sit behind the audio device; touching them resets that device and drops the whole session.
"""
import ctypes as C
import os

RES_MEDIUM = 1
VIDEO_RGB = 0
DEPTH_REGISTERED = 4
DEVICE_CAMERA = 0x02
WIDTH, HEIGHT = 640, 480


class FrameMode(C.Structure):
    _fields_ = [('reserved', C.c_uint32), ('resolution', C.c_int), ('format', C.c_int32), ('bytes', C.c_int32),
                ('width', C.c_int16), ('height', C.c_int16), ('data_bits', C.c_int8), ('padding_bits', C.c_int8),
                ('framerate', C.c_int8), ('is_valid', C.c_int8)]


Callback = C.CFUNCTYPE(None, C.c_void_p, C.c_void_p, C.c_uint32)


def _load(libdir):
    fn = C.CDLL(os.path.join(libdir, 'libfreenect.so.0'))
    fn.freenect_init.argtypes = [C.POINTER(C.c_void_p), C.c_void_p]
    fn.freenect_select_subdevices.argtypes = [C.c_void_p, C.c_int]
    fn.freenect_num_devices.argtypes = [C.c_void_p]
    fn.freenect_open_device.argtypes = [C.c_void_p, C.POINTER(C.c_void_p), C.c_int]
    fn.freenect_find_video_mode.restype = FrameMode
    fn.freenect_find_video_mode.argtypes = [C.c_int, C.c_int]
    fn.freenect_find_depth_mode.restype = FrameMode
    fn.freenect_find_depth_mode.argtypes = [C.c_int, C.c_int]
    fn.freenect_set_video_mode.argtypes = [C.c_void_p, FrameMode]
    fn.freenect_set_depth_mode.argtypes = [C.c_void_p, FrameMode]
    fn.freenect_set_video_callback.argtypes = [C.c_void_p, Callback]
    fn.freenect_set_depth_callback.argtypes = [C.c_void_p, Callback]
    for name in ('freenect_start_video', 'freenect_start_depth', 'freenect_stop_video', 'freenect_stop_depth',
                 'freenect_close_device'):
        getattr(fn, name).argtypes = [C.c_void_p]
    fn.freenect_process_events.argtypes = [C.c_void_p]
    fn.freenect_shutdown.argtypes = [C.c_void_p]
    fn.freenect_camera_to_world.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int, C.POINTER(C.c_double), C.POINTER(C.c_double)]
    return fn


class Kinect:
    """on_rgb(bytes, device_ts) and on_depth(bytes, device_ts) run inside process_events()."""

    def __init__(self, on_rgb, on_depth, libdir=os.path.expanduser('~/scanrig/local/lib'), index=0):
        self.fn = _load(libdir)
        self.ctx, self.dev = C.c_void_p(), C.c_void_p()
        if self.fn.freenect_init(C.byref(self.ctx), None) < 0:
            raise RuntimeError('freenect_init failed')
        self.fn.freenect_select_subdevices(self.ctx, DEVICE_CAMERA)
        if self.fn.freenect_num_devices(self.ctx) <= index:
            self.fn.freenect_shutdown(self.ctx)
            raise RuntimeError('no Kinect found')
        if self.fn.freenect_open_device(self.ctx, C.byref(self.dev), index) < 0:
            self.fn.freenect_shutdown(self.ctx)
            raise RuntimeError('cannot open the Kinect camera (in use, or missing udev permission?)')
        self.vmode = self.fn.freenect_find_video_mode(RES_MEDIUM, VIDEO_RGB)
        self.dmode = self.fn.freenect_find_depth_mode(RES_MEDIUM, DEPTH_REGISTERED)
        # keep the ctypes callbacks referenced for as long as the device can call them
        self._video_cb = Callback(lambda _d, data, ts: on_rgb(C.string_at(data, self.vmode.bytes), ts))
        self._depth_cb = Callback(lambda _d, data, ts: on_depth(C.string_at(data, self.dmode.bytes), ts))
        self.fn.freenect_set_video_mode(self.dev, self.vmode)
        self.fn.freenect_set_depth_mode(self.dev, self.dmode)
        self.fn.freenect_set_video_callback(self.dev, self._video_cb)
        self.fn.freenect_set_depth_callback(self.dev, self._depth_cb)
        self.started = False

    def focal_estimate(self):
        """Focal length (px) implied by libfreenect's registration model, a prior for calibration."""
        wx0, wx1, wy = C.c_double(), C.c_double(), C.c_double()
        self.fn.freenect_camera_to_world(self.dev, WIDTH // 2, HEIGHT // 2, 1000, C.byref(wx0), C.byref(wy))
        self.fn.freenect_camera_to_world(self.dev, WIDTH // 2 + 100, HEIGHT // 2, 1000, C.byref(wx1), C.byref(wy))
        return 100 * 1000 / (wx1.value - wx0.value) if wx1.value != wx0.value else None

    def start(self):
        self.fn.freenect_start_video(self.dev)
        self.fn.freenect_start_depth(self.dev)
        self.started = True

    def process_events(self):
        return self.fn.freenect_process_events(self.ctx) >= 0

    def close(self):
        if self.started:
            self.fn.freenect_stop_video(self.dev)
            self.fn.freenect_stop_depth(self.dev)
            self.started = False
        self.fn.freenect_close_device(self.dev)
        self.fn.freenect_shutdown(self.ctx)
