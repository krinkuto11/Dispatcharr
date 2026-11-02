"""Synthetic FFmpeg decoder probe for one live Dispatcharr output URL.

The probe detects pauses in decoded-media progress. It is not a substitute for
telemetry from a spectator's player buffer or network/device class.
"""

import os
from pathlib import Path
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


STALL_AFTER = float(os.getenv("QOE_PROBE_STALL_AFTER_SECONDS", "2"))
STARTUP_SLOW_AFTER = float(os.getenv("QOE_PROBE_STARTUP_SLOW_AFTER_SECONDS", "10"))
PORT = int(os.getenv("QOE_PROBE_PORT", "9194"))
URL_FILE = Path(os.getenv("QOE_PROBE_URL_FILE", "/run/secrets/qoe_probe_url"))
FFMPEG = os.getenv("QOE_PROBE_FFMPEG", "ffmpeg")

ERROR_PATTERNS = {
    "decode": re.compile(r"(?:error while decoding|decode error|corrupt|damaged macroblock)", re.I),
    "timestamp": re.compile(r"(?:non.monotonic|DTS|PTS|timestamp discontinuity)", re.I),
    "network": re.compile(r"(?:timed out|connection reset|broken pipe|HTTP error|server returned)", re.I),
}


class ProbeState:
    def __init__(self, stall_after=STALL_AFTER, startup_slow_after=STARTUP_SLOW_AFTER, clock=time.monotonic):
        self.lock = threading.Lock()
        self.clock = clock
        self.stall_after = stall_after
        self.startup_slow_after = startup_slow_after
        self.running = False
        self.process_started = None
        self.first_media_at = None
        self.last_media_at = None
        self.last_media_time_us = None
        self.stall_started = None
        self.stalls_total = 0
        self.stall_seconds_total = 0.0
        self.startup_slow_total = 0
        self.startup_seconds_last = 0.0
        self.timeline_resets_total = 0
        self.process_exits_total = 0
        self.last_exit_code = 0
        self.config_errors_total = 0
        self.errors = {category: 0 for category in ERROR_PATTERNS}

    def start_process(self):
        with self.lock:
            now = self.clock()
            self.running = True
            self.process_started = now
            self.first_media_at = None
            self.last_media_at = None
            self.last_media_time_us = None
            self.stall_started = None

    def media_progress(self, media_time_us):
        try:
            media_time_us = int(media_time_us)
        except (TypeError, ValueError):
            return
        with self.lock:
            now = self.clock()
            if not self.running:
                return
            previous = self.last_media_time_us
            if previous is not None and media_time_us == previous:
                self._tick_locked(now)
                return
            if previous is not None and media_time_us < previous:
                self.timeline_resets_total += 1
            self.last_media_time_us = media_time_us
            if self.first_media_at is None:
                self.first_media_at = now
                self.startup_seconds_last = now - self.process_started
                if self.startup_seconds_last >= self.startup_slow_after:
                    self.startup_slow_total += 1
            if self.stall_started is not None:
                self.stalls_total += 1
                self.stall_seconds_total += max(0.0, now - self.stall_started)
                self.stall_started = None
            self.last_media_at = now

    def _tick_locked(self, now):
        if self.running and self.first_media_at is not None and self.stall_started is None:
            if now - self.last_media_at >= self.stall_after:
                self.stall_started = self.last_media_at

    def tick(self):
        with self.lock:
            self._tick_locked(self.clock())

    def record_error(self, line):
        # Raw stderr and stream URLs never enter exported metrics or logs.
        with self.lock:
            for category, pattern in ERROR_PATTERNS.items():
                if pattern.search(line):
                    self.errors[category] += 1
                    return

    def stop_process(self, exit_code):
        with self.lock:
            now = self.clock()
            self._tick_locked(now)
            if self.stall_started is not None:
                self.stalls_total += 1
                self.stall_seconds_total += max(0.0, now - self.stall_started)
                self.stall_started = None
            if self.first_media_at is None and self.process_started is not None:
                startup = now - self.process_started
                self.startup_seconds_last = startup
                if startup >= self.startup_slow_after:
                    self.startup_slow_total += 1
            self.running = False
            self.process_exits_total += 1
            self.last_exit_code = int(exit_code) if exit_code is not None else -1

    def metrics(self):
        with self.lock:
            now = self.clock()
            self._tick_locked(now)
            lines = [
                "# TYPE dispatcharr_probe_up gauge",
                f"dispatcharr_probe_up {int(self.running)}",
                "# TYPE dispatcharr_probe_media_progress_age_seconds gauge",
                f"dispatcharr_probe_media_progress_age_seconds {max(0.0, now - self.last_media_at) if self.last_media_at is not None else 0}",
                "# TYPE dispatcharr_probe_media_progress_present gauge",
                f"dispatcharr_probe_media_progress_present {int(self.last_media_at is not None)}",
                "# TYPE dispatcharr_probe_stall_active gauge",
                f"dispatcharr_probe_stall_active {int(self.stall_started is not None)}",
                "# TYPE dispatcharr_probe_stalls_total counter",
                f"dispatcharr_probe_stalls_total {self.stalls_total}",
                "# TYPE dispatcharr_probe_stall_seconds_total counter",
                f"dispatcharr_probe_stall_seconds_total {self.stall_seconds_total}",
                "# TYPE dispatcharr_probe_startup_slow_total counter",
                f"dispatcharr_probe_startup_slow_total {self.startup_slow_total}",
                "# TYPE dispatcharr_probe_startup_seconds_last gauge",
                f"dispatcharr_probe_startup_seconds_last {self.startup_seconds_last}",
                "# TYPE dispatcharr_probe_timeline_resets_total counter",
                f"dispatcharr_probe_timeline_resets_total {self.timeline_resets_total}",
                "# TYPE dispatcharr_probe_process_exits_total counter",
                f"dispatcharr_probe_process_exits_total {self.process_exits_total}",
                "# TYPE dispatcharr_probe_last_exit_code gauge",
                f"dispatcharr_probe_last_exit_code {self.last_exit_code}",
                "# TYPE dispatcharr_probe_config_errors_total counter",
                f"dispatcharr_probe_config_errors_total {self.config_errors_total}",
                "# TYPE dispatcharr_probe_errors_total counter",
            ]
            lines.extend(
                f'dispatcharr_probe_errors_total{{category="{category}"}} {count}'
                for category, count in sorted(self.errors.items())
            )
            return "\n".join(lines) + "\n"


def _progress_reader(stream, state):
    block = {}
    for raw in stream:
        line = raw.strip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        block[name] = value
        if name == "progress":
            media_us = block.get("out_time_us") or block.get("out_time_ms")
            if media_us is not None:
                state.media_progress(media_us)
            block = {}


def _stderr_reader(stream, state):
    for line in stream:
        state.record_error(line)


def _run_probe(state):
    while True:
        # Read before every restart to support credential/token rotation.
        try:
            url = URL_FILE.read_text(encoding="utf-8").strip()
            if not url or not url.startswith(("http://", "https://")):
                raise ValueError("Probe URL file must contain one HTTP(S) URL")
        except (OSError, ValueError):
            with state.lock:
                state.config_errors_total += 1
            time.sleep(5)
            continue
        command = [
            FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "warning",
            "-stats_period", "0.5", "-progress", "pipe:1",
            "-rw_timeout", "20000000", "-i", url,
            "-map", "0:v:0?", "-map", "0:a:0?", "-f", "null", "-",
        ]
        state.start_process()
        try:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1,
            )
        except OSError:
            state.stop_process(-1)
            time.sleep(5)
            continue
        stderr_thread = threading.Thread(target=_stderr_reader, args=(process.stderr, state), daemon=True)
        stderr_thread.start()
        _progress_reader(process.stdout, state)
        exit_code = process.wait()
        stderr_thread.join(timeout=1)
        process.stdout.close()
        process.stderr.close()
        state.stop_process(exit_code)
        time.sleep(2)


def main():
    state = ProbeState()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/metrics":
                body = state.metrics().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            elif self.path == "/healthz":
                body = b"ok\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
            else:
                body = b"not found\n"
                self.send_response(404)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *_args):
            return

    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    threading.Thread(target=_run_probe, args=(state,), daemon=True).start()
    while True:
        state.tick()
        # The HTTP server runs in a thread so state.tick() is independent of
        # Prometheus scrapes and FFmpeg progress writes.
        time.sleep(0.25)


if __name__ == "__main__":
    main()
