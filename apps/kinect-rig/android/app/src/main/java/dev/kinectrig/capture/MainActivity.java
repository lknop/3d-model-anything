package dev.kinectrig.capture;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.media.AudioManager;
import android.media.ToneGenerator;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.Gravity;
import android.view.WindowManager;
import android.widget.FrameLayout;
import android.widget.TextView;

import com.google.ar.core.ArCoreApk;
import com.google.ar.core.Camera;
import com.google.ar.core.CameraConfig;
import com.google.ar.core.CameraConfigFilter;
import com.google.ar.core.CameraIntrinsics;
import com.google.ar.core.Frame;
import com.google.ar.core.Pose;
import com.google.ar.core.Session;
import com.google.ar.core.TrackingState;

import java.io.File;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

/**
 * Rig phone: ARCore tracking with a camera viewfinder. Streams every frame's camera pose to the
 * laptop (RigLink) and records sessions on the phone when the laptop says so (Recorder).
 * Poses use ARCore's world frame: meters, +Y up (gravity), camera looking down -Z in the
 * sensor's landscape orientation, the same convention as ARKit.
 */
public class MainActivity extends Activity implements GLSurfaceView.Renderer, RigLink.Commands {
    private static final String TAG = "KinectRig";

    private final BackgroundRenderer background = new BackgroundRenderer();
    private final RigLink link = new RigLink(this);
    private final ToneGenerator beep = new ToneGenerator(AudioManager.STREAM_NOTIFICATION, 80);
    private GLSurfaceView surface;
    private TextView status;
    private Session session;
    private String cameraConfig = "null";
    private Recorder recorder; // GL thread only
    private boolean installRequested;
    private long frames;
    private long lastStatusMs;
    private long lastFrameCount;
    private boolean showingHold;
    private int surfaceWidth;
    private int surfaceHeight;
    private int displayRotation = -1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        surface = new GLSurfaceView(this);
        surface.setPreserveEGLContextOnPause(true);
        surface.setEGLContextClientVersion(2);
        surface.setEGLConfigChooser(8, 8, 8, 8, 16, 0);
        surface.setRenderer(this);
        surface.setRenderMode(GLSurfaceView.RENDERMODE_CONTINUOUSLY);
        status = new TextView(this);
        status.setTextColor(Color.WHITE);
        status.setShadowLayer(4, 0, 0, Color.BLACK);
        status.setTextSize(16);
        status.setPadding(24, 16, 24, 16);
        FrameLayout root = new FrameLayout(this);
        root.addView(surface);
        root.addView(status, new FrameLayout.LayoutParams(FrameLayout.LayoutParams.WRAP_CONTENT,
                FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.TOP | Gravity.START));
        setContentView(root);
        link.start();
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, 0);
            return;
        }
        try {
            if (session == null) {
                if (ArCoreApk.getInstance().requestInstall(this, !installRequested) == ArCoreApk.InstallStatus.INSTALL_REQUESTED) {
                    installRequested = true;
                    return;
                }
                session = new Session(this);
                chooseCameraConfig(session);
            }
            session.resume();
        } catch (Exception e) {
            Log.e(TAG, "ARCore unavailable", e);
            status.setText("ARCore error: " + e.getMessage());
            session = null;
            return;
        }
        surface.onResume();
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (session != null) {
            surface.onPause();
            session.pause();
        }
    }

    @Override
    protected void onDestroy() {
        link.stop();
        beep.release();
        if (session != null) session.close();
        super.onDestroy();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] results) {
        if (results.length == 0 || results[0] != PackageManager.PERMISSION_GRANTED) {
            status.setText("Camera permission is required");
            finish();
        }
    }

    /** The largest CPU image ARCore offers at 30 fps: these images become the scan's photos. */
    private void chooseCameraConfig(Session s) {
        CameraConfigFilter filter = new CameraConfigFilter(s)
                .setTargetFps(EnumSet.of(CameraConfig.TargetFps.TARGET_FPS_30));
        List<CameraConfig> configs = s.getSupportedCameraConfigs(filter);
        CameraConfig best = null;
        StringBuilder offered = new StringBuilder();
        for (CameraConfig c : configs) {
            offered.append(offered.length() == 0 ? "" : ",").append('"').append(describe(c)).append('"');
            if (best == null || area(c.getImageSize()) > area(best.getImageSize())) best = c;
        }
        if (best == null) return;
        s.setCameraConfig(best);
        cameraConfig = String.format(Locale.US, "{\"chosen\":\"%s\",\"offered\":[%s]}", describe(best), offered);
        Log.i(TAG, "camera config " + cameraConfig);
    }

    private static String describe(CameraConfig c) {
        return String.format(Locale.US, "image %dx%d texture %dx%d fps %s depth %s id %s",
                c.getImageSize().getWidth(), c.getImageSize().getHeight(),
                c.getTextureSize().getWidth(), c.getTextureSize().getHeight(),
                c.getFpsRange(), c.getDepthSensorUsage(), c.getCameraId());
    }

    private static long area(android.util.Size s) {
        return (long) s.getWidth() * s.getHeight();
    }

    // --- laptop commands (arrive on the link thread; recording state lives on the GL thread) ---

    @Override
    public void onStart(String name) {
        String safe = name.replaceAll("[^A-Za-z0-9_-]", "_");
        surface.queueEvent(() -> {
            if (recorder != null) {
                link.reply("{\"type\":\"ack\",\"cmd\":\"start\",\"ok\":false,\"error\":\"already recording " + recorder.name + "\"}");
                return;
            }
            try {
                recorder = new Recorder(new File(getExternalFilesDir(null), "sessions"), safe, cameraConfig);
                link.reply("{\"type\":\"ack\",\"cmd\":\"start\",\"ok\":true,\"dir\":\"" + recorder.dir.getAbsolutePath() + "\"}");
            } catch (Exception e) {
                link.reply("{\"type\":\"ack\",\"cmd\":\"start\",\"ok\":false,\"error\":\"" + e.getMessage() + "\"}");
            }
        });
    }

    @Override
    public void onStop() {
        surface.queueEvent(() -> {
            if (recorder == null) {
                link.reply("{\"type\":\"ack\",\"cmd\":\"stop\",\"ok\":false,\"error\":\"not recording\"}");
                return;
            }
            Recorder r = recorder;
            recorder = null;
            try {
                link.reply("{\"type\":\"ack\",\"cmd\":\"stop\",\"ok\":true,\"dir\":\"" + r.dir.getAbsolutePath() + "\",\"summary\":" + r.stop() + "}");
            } catch (Exception e) {
                link.reply("{\"type\":\"ack\",\"cmd\":\"stop\",\"ok\":false,\"error\":\"" + e.getMessage() + "\"}");
            }
        });
    }

    // --- rendering ---

    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        GLES20.glClearColor(0f, 0f, 0f, 1f);
        background.createOnGlThread();
    }

    @Override
    public void onSurfaceChanged(GL10 gl, int width, int height) {
        GLES20.glViewport(0, 0, width, height);
        surfaceWidth = width;
        surfaceHeight = height;
        displayRotation = -1; // re-sent on the next frame
    }

    @Override
    public void onDrawFrame(GL10 gl) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);
        if (session == null) return;
        // A 180-degree flip keeps the surface size, so onSurfaceChanged doesn't see it.
        int rotation = getDisplay().getRotation();
        if (rotation != displayRotation) {
            displayRotation = rotation;
            session.setDisplayGeometry(rotation, surfaceWidth, surfaceHeight);
        }
        try {
            session.setCameraTextureName(background.textureId());
            Frame frame = session.update();
            background.draw(frame);
            Camera camera = frame.getCamera();
            frames++;
            link.send(frameLine(frame, camera));
            if (recorder != null) {
                String keyframe = recorder.onFrame(frame, camera);
                if (keyframe != null) {
                    link.send(keyframe);
                    beep.startTone(ToneGenerator.TONE_PROP_BEEP, 60);
                }
            }
            updateStatus(camera);
        } catch (Exception e) {
            Log.e(TAG, "frame failed", e);
        }
    }

    private String frameLine(Frame frame, Camera camera) {
        Pose p = camera.getPose();
        StringBuilder line = new StringBuilder(256);
        line.append(String.format(Locale.US,
                "{\"type\":\"frame\",\"n\":%d,\"t_ns\":%d,\"phone_ns\":%d,\"tracking\":\"%s\",\"reason\":\"%s\","
                        + "\"pose\":[%.5f,%.5f,%.5f,%.6f,%.6f,%.6f,%.6f]",
                frames, frame.getTimestamp(), SystemClock.elapsedRealtimeNanos(), camera.getTrackingState(),
                camera.getTrackingFailureReason(), p.tx(), p.ty(), p.tz(), p.qx(), p.qy(), p.qz(), p.qw()));
        if (frames % 30 == 1) { // intrinsics of the CPU image, now and then
            CameraIntrinsics k = camera.getImageIntrinsics();
            float[] f = k.getFocalLength();
            float[] c = k.getPrincipalPoint();
            int[] d = k.getImageDimensions();
            line.append(String.format(Locale.US, ",\"intrinsics\":{\"fx\":%.3f,\"fy\":%.3f,\"cx\":%.3f,\"cy\":%.3f,\"width\":%d,\"height\":%d}",
                    f[0], f[1], c[0], c[1], d[0], d[1]));
        }
        return line.append('}').toString();
    }

    private void updateStatus(Camera camera) {
        long now = SystemClock.uptimeMillis();
        boolean hold = recorder != null && recorder.holdStill();
        if (hold != showingHold) { // the pause prompt can't wait for the next refresh
            showingHold = hold;
            lastStatusMs = 0;
        }
        if (now - lastStatusMs < 500) return;
        double fps = (frames - lastFrameCount) * 1000.0 / Math.max(1, now - lastStatusMs);
        lastStatusMs = now;
        lastFrameCount = frames;
        TrackingState state = camera.getTrackingState();
        String rec = recorder == null ? "" : String.format(Locale.US, "\n● REC %s  %d keyframes%s",
                recorder.name, recorder.keyframes(), hold ? "\n✋ HOLD STILL" : "");
        String text = String.format(Locale.US, "%s%s  %.0f fps  laptop %s%s",
                state, state == TrackingState.TRACKING ? "" : " (" + camera.getTrackingFailureReason() + ")",
                fps, link.connected() ? "connected" : "waiting on :" + RigLink.PORT, rec);
        runOnUiThread(() -> {
            status.setText(text);
            status.setTextColor(recorder == null ? Color.WHITE : Color.rgb(255, 120, 120));
        });
    }
}
