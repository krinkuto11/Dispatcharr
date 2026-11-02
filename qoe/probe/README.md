# Synthetic Dispatcharr decoder probe

This container opens **one real stream connection** and decodes video/audio
with FFmpeg. It reports pauses in decoded-media progress, startup delay,
process exits and bounded decoder/network errors on `GET /metrics` (port 9194).
It is a synthetic decoder: it does not measure the buffer of a spectator's
specific app/device, so correlate it with client-side QoE telemetry when
available. Run one instance per channel/output format/profile worth monitoring.

## Run in a test environment

1. Put one authorized Dispatcharr HTTP(S) stream URL in a file readable by the
   container user (UID 10001). Keep the file outside the repository. The URL
   never appears in metrics or probe logs, but it is visible in the FFmpeg
   process argument list inside the container.
2. Build with `docker build -t dispatcharr-qoe-probe .`.
3. Run on a network that can reach Dispatcharr and that represents the viewers
   being studied. For example:

   ```sh
   docker run --rm --name dispatcharr-qoe-probe \
     --network host --read-only --tmpfs /tmp \
     --mount type=bind,src=/path/to/probe-url.txt,dst=/run/secrets/qoe_probe_url,readonly \
     dispatcharr-qoe-probe
   ```

4. Scrape `http://<probe-host>:9194/metrics` every 5 seconds. Restrict this
   port to Prometheus/NetBird; the endpoint has no authentication. Add stable
   Prometheus target labels for channel and format, never the URL or token.

`QOE_PROBE_STALL_AFTER_SECONDS` defaults to 2; `QOE_PROBE_STARTUP_SLOW_AFTER_SECONDS`
defaults to 10. The probe retries after an FFmpeg exit and rereads the URL file,
so a rotated token is picked up. It treats a backwards media timestamp as a
timeline reset and progress, recording `dispatcharr_probe_timeline_resets_total`.

PromQL examples:

```promql
increase(dispatcharr_probe_stalls_total[5m])
increase(dispatcharr_probe_stall_seconds_total[5m])
dispatcharr_probe_stall_active
dispatcharr_probe_media_progress_age_seconds
```

Test locally with `python -m unittest discover -s . -p test_probe.py`. Before
using a live production URL, validate the probe on a controlled stream where
media stops for 3–5 seconds and resumes, and compare the detected gap with a
real player. Keep the probe's connection in mind when interpreting audience
counts and channel teardown.
