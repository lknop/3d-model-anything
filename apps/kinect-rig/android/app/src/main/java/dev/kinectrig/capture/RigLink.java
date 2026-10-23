package dev.kinectrig.capture;

import android.os.SystemClock;
import android.util.Log;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

/**
 * Line-oriented link to the laptop over TCP (reach it with `adb forward tcp:8765 tcp:8765`).
 *
 * Phone → laptop: one JSON object per line (frames, keyframes, pongs, acks).
 * Laptop → phone:
 *   "ping <id>"     answered at once with {"type":"pong","id":<id>,"phone_ns":<elapsedRealtimeNanos>},
 *                   so the laptop can estimate the clock offset from the lowest round trip
 *   "start <name>"  begin recording a session on the phone
 *   "stop"          end it; the activity replies with an ack carrying the session summary
 * One client at a time; streamed lines are dropped while nobody is connected.
 */
final class RigLink {
    static final int PORT = 8765;
    private static final String TAG = "RigLink";

    interface Commands {
        void onStart(String name);

        void onStop();
    }

    private final LinkedBlockingQueue<String> outbox = new LinkedBlockingQueue<>(600);
    private final Commands commands;
    private volatile Socket client;
    private volatile boolean running;
    private ServerSocket server;
    private Thread acceptThread;

    RigLink(Commands commands) {
        this.commands = commands;
    }

    boolean connected() {
        return client != null;
    }

    void start() {
        running = true;
        acceptThread = new Thread(this::acceptLoop, "link-accept");
        acceptThread.start();
    }

    void stop() {
        running = false;
        closeQuietly(server);
        closeQuietly(client);
        if (acceptThread != null) acceptThread.interrupt();
    }

    /** Queue a line for the connected client; dropped when nobody listens or the link is behind. */
    void send(String line) {
        if (client != null && !outbox.offer(line)) {
            Log.w(TAG, "outbox full, dropping a line");
        }
    }

    /** Write a line now, ahead of anything queued (pongs, acks). */
    void reply(String line) {
        Socket s = client;
        if (s == null) return;
        try {
            write(s.getOutputStream(), line);
        } catch (Exception e) {
            Log.i(TAG, "reply failed: " + e.getMessage());
        }
    }

    private synchronized void write(OutputStream out, String line) throws java.io.IOException {
        out.write((line + "\n").getBytes(StandardCharsets.UTF_8));
        out.flush();
    }

    private void acceptLoop() {
        try {
            server = new ServerSocket(PORT);
            while (running) {
                Socket s = server.accept();
                s.setTcpNoDelay(true);
                outbox.clear();
                client = s;
                Thread reader = new Thread(() -> readLoop(s), "link-read");
                reader.start();
                writeLoop(s);
                client = null;
                closeQuietly(s);
                reader.interrupt();
            }
        } catch (Exception e) {
            if (running) Log.e(TAG, "server stopped", e);
        }
    }

    private void writeLoop(Socket s) {
        try {
            OutputStream out = s.getOutputStream();
            while (running && !s.isClosed()) {
                String line = outbox.poll(500, TimeUnit.MILLISECONDS);
                if (line != null) write(out, line);
            }
        } catch (Exception e) {
            Log.i(TAG, "client gone: " + e.getMessage());
        }
    }

    private void readLoop(Socket s) {
        try {
            BufferedReader in = new BufferedReader(new InputStreamReader(s.getInputStream(), StandardCharsets.UTF_8));
            String line;
            while ((line = in.readLine()) != null) {
                line = line.trim();
                if (line.startsWith("ping ")) {
                    long now = SystemClock.elapsedRealtimeNanos();
                    // Bypass the frame queue so the round trip measures the link, not the backlog.
                    write(s.getOutputStream(), "{\"type\":\"pong\",\"id\":" + Long.parseLong(line.substring(5).trim()) + ",\"phone_ns\":" + now + "}");
                } else if (line.startsWith("start ")) {
                    commands.onStart(line.substring(6).trim());
                } else if (line.equals("stop")) {
                    commands.onStop();
                }
            }
        } catch (Exception e) {
            Log.i(TAG, "reader done: " + e.getMessage());
        }
        closeQuietly(s);
    }

    private static void closeQuietly(java.io.Closeable c) {
        try {
            if (c != null) c.close();
        } catch (Exception ignored) {
        }
    }
}
