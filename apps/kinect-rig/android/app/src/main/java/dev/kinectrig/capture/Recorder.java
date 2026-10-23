package dev.kinectrig.capture;

import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.os.SystemClock;
import android.util.Log;

import com.google.ar.core.Camera;
import com.google.ar.core.CameraIntrinsics;
import com.google.ar.core.Frame;
import com.google.ar.core.ImageMetadata;
import com.google.ar.core.Pose;
import com.google.ar.core.TrackingState;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * One recording on the phone, in <files>/sessions/<name>/:
 *
 *   poses.jsonl      every camera frame: timestamps, tracking state, camera-to-world pose
 *   keyframes.jsonl  frames saved as images: pose, CPU-image intrinsics, exposure, angular speed
 *   images/NNNNNN.jpg  the CPU camera image in sensor orientation (landscape even on a portrait rig), which the intrinsics describe
 *   session.json     camera config and counts, written on stop
 *
 * Called on the GL thread; JPEG encoding runs on its own thread so tracking keeps its frame rate.
 */
final class Recorder {
    private static final String TAG = "Recorder";
    // A keyframe is due after this much new viewpoint...
    private static final long MIN_INTERVAL_NS = 300_000_000L;
    private static final long MAX_INTERVAL_NS = 3_000_000_000L; // ...or time, while standing still
    private static final float MOVE_M = 0.25f;
    private static final float TURN_DEG = 12f;
    // ...and is taken on the first frame whose predicted motion blur is small enough. Indoor
    // exposures are 20-30 ms, so "sharp" means the operator pauses; the bar drops if they don't.
    private static final float SHARP_PX = 2.5f;
    private static final float RELAXED_PX = 6f;
    private static final long RELAX_NS = 1_500_000_000L;
    private static final long GIVE_UP_NS = 4_000_000_000L;
    private static final float TYPICAL_DEPTH_M = 2f; // scales translation into image motion
    private static final long DEFAULT_EXPOSURE_NS = 20_000_000L;
    private static final int MAX_PENDING = 4;

    final String name;
    final File dir;
    private final String cameraConfig;
    private final BufferedWriter poses;
    private final BufferedWriter keyframes;
    private final ExecutorService encoder = Executors.newSingleThreadExecutor();
    private final AtomicInteger pending = new AtomicInteger();
    private final AtomicInteger written = new AtomicInteger();
    private final long startedNs = SystemClock.elapsedRealtimeNanos();
    private int frames;
    private int keyframeCount;
    private int skippedBusy;
    private Pose lastPose;
    private long lastPoseNs;
    private Pose lastKeyPose;
    private long lastKeyNs;
    private long dueSinceNs = -1;
    private float turnDps;
    private float speedMps;
    private float blurPx;

    Recorder(File root, String name, String cameraConfig) throws IOException {
        this.name = name;
        this.cameraConfig = cameraConfig;
        dir = new File(root, name);
        File images = new File(dir, "images");
        if (!images.mkdirs() && !images.isDirectory()) throw new IOException("cannot create " + images);
        poses = new BufferedWriter(new FileWriter(new File(dir, "poses.jsonl")), 1 << 16);
        keyframes = new BufferedWriter(new FileWriter(new File(dir, "keyframes.jsonl")), 1 << 14);
    }

    int keyframes() {
        return keyframeCount;
    }

    /** A keyframe is due but the rig is moving too much for a sharp one: the operator should pause. */
    boolean holdStill() {
        return dueSinceNs >= 0;
    }

    /** Record one ARCore frame; returns a keyframe line for the laptop when this frame became one. */
    String onFrame(Frame frame, Camera camera) throws IOException {
        long t = frame.getTimestamp();
        Pose pose = camera.getPose();
        TrackingState state = camera.getTrackingState();
        if (lastPose != null && t > lastPoseNs) {
            float dt = (t - lastPoseNs) / 1e9f;
            turnDps = angleDeg(lastPose, pose) / dt;
            speedMps = distance(lastPose, pose) / dt;
        }
        lastPose = pose;
        lastPoseNs = t;
        frames++;
        poses.write(String.format(Locale.US,
                "{\"t_ns\":%d,\"phone_ns\":%d,\"tracking\":\"%s\",\"pose\":[%.5f,%.5f,%.5f,%.6f,%.6f,%.6f,%.6f]}\n",
                t, SystemClock.elapsedRealtimeNanos(), state, pose.tx(), pose.ty(), pose.tz(), pose.qx(), pose.qy(), pose.qz(), pose.qw()));

        if (state != TrackingState.TRACKING || !due(t, pose)) {
            dueSinceNs = -1;
            return null;
        }
        if (dueSinceNs < 0) dueSinceNs = t;
        long exposureNs = exposureNs(frame);
        float fx = camera.getImageIntrinsics().getFocalLength()[0];
        blurPx = (float) ((Math.toRadians(turnDps) + speedMps / TYPICAL_DEPTH_M) * fx * exposureNs / 1e9);
        long waited = t - dueSinceNs;
        float allowed = waited >= GIVE_UP_NS ? Float.MAX_VALUE : waited >= RELAX_NS ? RELAXED_PX : SHARP_PX;
        if (blurPx > allowed) return null;
        if (pending.get() >= MAX_PENDING) {
            skippedBusy++;
            return null;
        }
        Yuv yuv;
        try (Image image = frame.acquireCameraImage()) {
            yuv = new Yuv(image);
        } catch (Exception e) { // NotYetAvailableException, ResourceExhaustedException
            return null;
        }
        int index = keyframeCount++;
        lastKeyPose = pose;
        lastKeyNs = t;
        dueSinceNs = -1;
        String file = String.format(Locale.US, "images/%06d.jpg", index);
        CameraIntrinsics k = camera.getImageIntrinsics();
        float[] f = k.getFocalLength();
        float[] c = k.getPrincipalPoint();
        String line = String.format(Locale.US,
                "{\"i\":%d,\"file\":\"%s\",\"t_ns\":%d,\"pose\":[%.5f,%.5f,%.5f,%.6f,%.6f,%.6f,%.6f],"
                        + "\"intrinsics\":{\"fx\":%.3f,\"fy\":%.3f,\"cx\":%.3f,\"cy\":%.3f,\"width\":%d,\"height\":%d},"
                        + "\"turn_dps\":%.2f,\"speed_mps\":%.3f,\"blur_px\":%.2f,\"waited_ms\":%d%s}",
                index, file, t, pose.tx(), pose.ty(), pose.tz(), pose.qx(), pose.qy(), pose.qz(), pose.qw(),
                f[0], f[1], c[0], c[1], yuv.width, yuv.height, turnDps, speedMps, blurPx, waited / 1_000_000, exposure(frame));
        keyframes.write(line + "\n");
        pending.incrementAndGet();
        File out = new File(dir, file);
        encoder.execute(() -> {
            try (FileOutputStream os = new FileOutputStream(out)) {
                new YuvImage(yuv.toNv21(), ImageFormat.NV21, yuv.width, yuv.height, null)
                        .compressToJpeg(new Rect(0, 0, yuv.width, yuv.height), 92, os);
                written.incrementAndGet();
            } catch (IOException e) {
                Log.e(TAG, "keyframe write failed", e);
            } finally {
                pending.decrementAndGet();
            }
        });
        return "{\"type\":\"keyframe\"," + line.substring(1);
    }

    /** Finish writing; returns the session summary (also saved as session.json). */
    String stop() throws IOException {
        encoder.shutdown();
        try {
            encoder.awaitTermination(30, TimeUnit.SECONDS);
        } catch (InterruptedException ignored) {
            Thread.currentThread().interrupt();
        }
        poses.close();
        keyframes.close();
        String summary = String.format(Locale.US,
                "{\"name\":\"%s\",\"format\":\"kinect-rig-phone/1\",\"frames\":%d,\"keyframes\":%d,\"keyframesWritten\":%d,"
                        + "\"skippedBusy\":%d,\"seconds\":%.1f,\"camera\":%s,"
                        + "\"conventions\":\"ARCore world: meters, +Y up (gravity), right-handed; pose = camera-to-world "
                        + "translation + quaternion (x,y,z,w); camera looks down -Z, +Y up in sensor (landscape) orientation; "
                        + "t_ns = camera timestamp, phone_ns = SystemClock.elapsedRealtimeNanos\"}",
                name, frames, keyframeCount, written.get(), skippedBusy,
                (SystemClock.elapsedRealtimeNanos() - startedNs) / 1e9, cameraConfig);
        try (FileWriter w = new FileWriter(new File(dir, "session.json"))) {
            w.write(summary + "\n");
        }
        return summary;
    }

    /** Enough new viewpoint (or time) since the last keyframe. */
    private boolean due(long t, Pose pose) {
        if (lastKeyPose == null) return true;
        long dt = t - lastKeyNs;
        if (dt < MIN_INTERVAL_NS) return false;
        return distance(lastKeyPose, pose) >= MOVE_M || angleDeg(lastKeyPose, pose) >= TURN_DEG || dt >= MAX_INTERVAL_NS;
    }

    private static long exposureNs(Frame frame) {
        try {
            return frame.getImageMetadata().getLong(ImageMetadata.SENSOR_EXPOSURE_TIME);
        } catch (Exception e) { // metadata not offered on every device
            return DEFAULT_EXPOSURE_NS;
        }
    }

    private static String exposure(Frame frame) {
        StringBuilder s = new StringBuilder();
        try {
            ImageMetadata m = frame.getImageMetadata();
            try {
                long exposure = m.getLong(ImageMetadata.SENSOR_EXPOSURE_TIME);
                s.append(",\"exposure_ns\":").append(exposure);
            } catch (Exception ignored) {
            }
            try {
                int iso = m.getInt(ImageMetadata.SENSOR_SENSITIVITY);
                s.append(",\"iso\":").append(iso);
            } catch (Exception ignored) {
            }
        } catch (Exception ignored) {
        }
        return s.toString();
    }

    private static float distance(Pose a, Pose b) {
        float[] p = a.getTranslation();
        float[] q = b.getTranslation();
        return (float) Math.sqrt(sq(p[0] - q[0]) + sq(p[1] - q[1]) + sq(p[2] - q[2]));
    }

    private static float angleDeg(Pose a, Pose b) {
        float[] qa = a.getRotationQuaternion();
        float[] qb = b.getRotationQuaternion();
        float dot = Math.abs(qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3]);
        return (float) Math.toDegrees(2 * Math.acos(Math.min(1f, dot)));
    }

    private static float sq(float v) {
        return v * v;
    }

    /** A copy of a YUV_420_888 camera image, so the Image can go back to ARCore immediately. */
    private static final class Yuv {
        final int width;
        final int height;
        final byte[] y;
        final byte[] u;
        final byte[] v;
        final int yRowStride;
        final int uvRowStride;
        final int uvPixelStride;

        Yuv(Image image) {
            width = image.getWidth();
            height = image.getHeight();
            Image.Plane[] planes = image.getPlanes();
            y = bytes(planes[0].getBuffer());
            u = bytes(planes[1].getBuffer());
            v = bytes(planes[2].getBuffer());
            yRowStride = planes[0].getRowStride();
            uvRowStride = planes[1].getRowStride();
            uvPixelStride = planes[1].getPixelStride();
        }

        byte[] toNv21() {
            byte[] out = new byte[width * height * 3 / 2];
            for (int row = 0; row < height; row++) {
                System.arraycopy(y, row * yRowStride, out, row * width, width);
            }
            int pos = width * height;
            for (int row = 0; row < height / 2; row++) {
                int base = row * uvRowStride;
                for (int col = 0; col < width / 2; col++) {
                    int i = base + col * uvPixelStride;
                    out[pos++] = v[i];
                    out[pos++] = u[i];
                }
            }
            return out;
        }

        private static byte[] bytes(ByteBuffer buffer) {
            byte[] b = new byte[buffer.remaining()];
            buffer.get(b);
            return b;
        }
    }
}
