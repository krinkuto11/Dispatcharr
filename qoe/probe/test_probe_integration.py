"""End-to-end decoder credit check against a local MPEG-TS HTTP stream."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import unittest

from probe import ProbeState, _progress_reader, _stderr_reader


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required")
class ProbeIntegrationTests(unittest.TestCase):
    def test_buffered_media_covers_a_midstream_http_pause(self):
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory) / "sample.ts"
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                "-i", "testsrc2=size=160x90:rate=25", "-t", "25",
                "-c:v", "mpeg2video", "-b:v", "500k", "-f", "mpegts",
                str(media),
            ], check=True, timeout=30)
            payload = media.read_bytes()

            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):
                    self.send_response(200)
                    self.send_header("Content-Type", "video/mp2t")
                    self.end_headers()
                    split = int(len(payload) * 0.75)
                    self.wfile.write(payload[:split])
                    self.wfile.flush()
                    time.sleep(4.5)
                    try:
                        self.wfile.write(payload[split:])
                        self.wfile.flush()
                    except BrokenPipeError:
                        pass

                def log_message(self, *_args):
                    pass

            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()
            try:
                state = ProbeState(stall_after=2.0)
                state.start_process()
                process = subprocess.Popen([
                    "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "warning",
                    "-stats_period", "0.5", "-progress", "pipe:1",
                    "-rw_timeout", "20000000", "-i",
                    f"http://127.0.0.1:{server.server_port}/stream",
                    "-map", "0:v:0?", "-f", "null", "-",
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                reader = threading.Thread(target=_progress_reader, args=(process.stdout, state), daemon=True)
                errors = threading.Thread(target=_stderr_reader, args=(process.stderr, state), daemon=True)
                reader.start()
                errors.start()
                deadline = time.monotonic() + 20
                while process.poll() is None and time.monotonic() < deadline:
                    state.tick()
                    time.sleep(0.1)
                if process.poll() is None:
                    process.kill()
                exit_code = process.wait(timeout=2)
                reader.join(timeout=2)
                errors.join(timeout=2)
                process.stdout.close()
                process.stderr.close()
                state.stop_process(exit_code)
                self.assertEqual(exit_code, 0)
                # The decoder already received ~18 seconds of media before
                # the 4.5-second network pause. That is not a playback stall.
                self.assertEqual(state.stalls_total, 0)
            finally:
                server.shutdown()
                server.server_close()
