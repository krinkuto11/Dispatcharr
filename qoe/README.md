# Dispatcharr QoE fork

Development base: upstream `dev` commit
`7bf5cff4be4697c629faaf267575761e48142705` (2026-10-30). The branch
`feat/qoe-observability` is cut directly from that commit; QoE changes are
separate commits. This base includes plugin v3, persistent plugin services and
the output-profile/HLS discontinuity work absent from stable `v0.33.0`.

## What the fork adds

- Bounded FFmpeg input/output stderr categories, output-profile process
  replacements, unexpected stops and chunks skipped by its input writer.
- Server-side delivery gaps above normal chunk cadence (with a two-second
  floor), separated into startup and playback, and chunks skipped by TS/fMP4
  clients after buffer expiry.
- HLS stale/malformed/cold-start failures, expired segments, segments still
  advertised but missing, and segment publication cadence. Successful
  steady-state HLS requests do not add Redis writes.
- [`iptv_qoe_exporter`](iptv_qoe_exporter/README.md), whose source is in this
  repo and exports those Redis counters with bounded labels.
- A [synthetic decoder probe](probe/README.md) that opens one real stream and
  reports pauses in decoded-media progress. It still cannot observe a
  spectator player's buffer or device.

No URL, token, client ID, IP or raw FFmpeg log line is used as a Prometheus
label. Core writes Redis only on a diagnostic event or once per HLS segment.
All metric writes are best effort and cannot abort a stream.

## Local verification

```sh
DJANGO_SECRET_KEY=qoe-test-only .venv/bin/python manage.py test \
  apps.proxy.live_proxy.tests.test_delivery_diagnostics \
  apps.proxy.live_proxy.tests.test_delivery_generators \
  apps.proxy.live_proxy.tests.test_delivery_hls_events \
  apps.proxy.live_proxy.tests.test_ffmpeg_diagnostics \
  apps.proxy.live_proxy.tests.test_output_buffer_missing \
  apps.proxy.live_proxy.tests.test_hls_views \
  apps.proxy.live_proxy.tests.test_output_profile_restart \
  apps.proxy.live_proxy.tests.test_stream_ts_client_registration \
  apps.proxy.live_proxy.tests.test_ts_generator_release --noinput
python -m unittest discover -s qoe/iptv_qoe_exporter -p 'test_*.py'
python -m unittest discover -s qoe/probe -p 'test_probe*.py'
```

GitHub Actions runs these checks and builds both images in
`.github/workflows/qoe-dev-ci.yml`. A local full image build requires a Docker
daemon and the upstream public `:base` image.

## Release path

The development branch remains pinned to the SHA above while core and plugin
signals are tested. When an upstream stable release contains that SHA (or a
verified equivalent tree), replay only the QoE commits onto its tag, add a
`.qoe-upstream-tag` marker and put the release-sync workflow on the fork's
default branch. After that, each published stable release produces a reviewed
candidate PR; a merge conflict leaves the deployed digest unchanged. The
workflow is intentionally guarded while the marker is absent.

Promote images by immutable digest after TS/HLS/fMP4 smoke tests and a decoder
probe comparison. The developer branch and scheduled workflow do not deploy
or restart production services.
