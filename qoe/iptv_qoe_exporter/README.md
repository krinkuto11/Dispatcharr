# Dispatcharr QoE Metrics plugin

Local implementation for filling the server-side observability gaps found in
the production Dispatcharr/OpenStream stack. It is **not installed or enabled
in production** by this workspace change.

**Compatibility gate:** this implementation targets the Dispatcharr `dev`
plugin API and output-buffer layout pinned in this fork. Stable `v0.33.0` has
neither `persistent_service` nor the same per-output buffer keys. Install it
only on a tested image built from this fork or a future compatible release.
The release/fork plan is in [`qoe/README.md`](../README.md).

## What it collects

- Live-channel state, client-set size, input profile, input bytes, FFmpeg speed,
  FPS, output bitrate and the age of the latest FFmpeg stats in Redis.
- Client counts and current egress rate grouped by output profile, plus
  audience grouped by bounded output format (`mpegts`, `hls`, `fmp4`,
  `unknown`). Client IDs, IP addresses, usernames and user agents are never
  exported.
- Durable counters for `channel_buffering`, `channel_failover`,
  `channel_reconnect`, `channel_error`, `stream_switch`, channel lifecycle and
  client connect/disconnect events. Reasons are reduced to a fixed allowlist.
- Recent FFmpeg `-progress` files created by the `OpenStream TimeControl`
  input profile (`/tmp/omts_<channelId>.prog`): speed, file update time and raw
  output timeline, FPS, frames, duplicated frames and dropped frames when the
  running FFmpeg build includes those keys. Dispatcharr substitutes its
  numeric database channel ID in that filename; the plugin maps it back to the
  channel UUID using the DB. The timeline includes the profile's
  `-output_ts_offset`, so it must not be interpreted as end-to-end latency.
  FFmpeg's dropped/duplicated-frame counters describe output-frame processing;
  with the active `-c:v copy` profile they may stay at zero and do not count all
  discarded input TS packets.
- Optional Dispatcharr core instrumentation parses FFmpeg stderr for both
  input and output-profile processes. When that patched image is installed, it
  exports bounded error categories (MPEG-TS continuity, corrupt/dropped
  packets, decode/timestamp/input/mux/I/O/other), last-error time, and
  output-profile progress (speed, FPS, bitrate and frame number), successful
  output-profile replacements, unexpected output reader/process stops and
  input chunks skipped by the output-profile writer. It stores counters and
  fixed categories in Redis, never raw stderr.
- The QoE fork records server delivery gaps of at least two seconds by
  startup/playback phase and recovery outcome, TS/fMP4 chunks skipped after
  buffer expiry, HLS errors and segment publication gaps. Episodes are counted
  when they recover or the connection ends; the server still cannot see the
  spectator player's own buffer.
- Current input and output buffer discontinuity boundaries, which can indicate
  a source switch or an output-profile restart. These are buffer-retained
  counts, not a cumulative packet-loss counter.
- Input and per-output-profile buffer freshness, retained chunk count and the
  timestamp span of retained chunks. These come from Redis indexes that
  Dispatcharr already writes; they indicate whether the server is still
  producing data, but do not measure a viewer's local playback buffer.
- Explicit metadata-present, stats-present, Redis-up and database-up gauges so
  a missing channel series can be distinguished from a scrape failure or an
  inactive channel.

Event counters are written to SQLite under Dispatcharr's plugin `data_dir`.
That directory is on the production `/data` volume, so counters survive plugin
and container restarts. This assumes the current AIO layout, where plugin
workers share that volume. If Dispatcharr is later split across hosts, move the
counter store to shared Redis or another shared database.

## Endpoint

- `GET /metrics`
- `GET /healthz`
- Default listen address: `0.0.0.0:9193`
- Prometheus scrape interval recommendation: `5s`

The HTTP endpoint has no authentication. Restrict access to the Prometheus
host/network (NetBird/firewall); do not publish port 9193 to the public
Internet.

## First PromQL checks

```promql
# Buffering, failover and error events by channel and bounded cause
sum by (channel_uuid, event, reason) (
  increase(dispatcharr_qoe_dispatcharr_events_total[5m])
)

# Active channels whose input FFmpeg stats have gone stale
dispatcharr_qoe_channel_active == 1
  and on (channel_uuid, channel_number)
dispatcharr_qoe_channel_ffmpeg_stats_present == 0

# Active-channel audience that is present on a configured channel with no clients
dispatcharr_qoe_channel_active == 1
  and on (channel_uuid, channel_number)
dispatcharr_qoe_channel_clients == 0

# Current egress rate split by output profile
sum by (output_profile_id) (dispatcharr_qoe_output_profile_egress_bytes_per_second)

# New server-side FFmpeg diagnostics in the last five minutes
sum by (channel_uuid, scope, output_profile_id, error_type) (
  increase(dispatcharr_qoe_ffmpeg_errors_total[5m])
)

# Output profile replacements, unexpected stops, and writer skips
sum by (channel_uuid, output_profile_id, reason) (
  increase(dispatcharr_qoe_output_profile_ffmpeg_restarts_total[5m])
)
sum by (channel_uuid, output_profile_id, reason) (
  increase(dispatcharr_qoe_output_profile_ffmpeg_unexpected_stops_total[5m])
)
sum by (channel_uuid, output_profile_id) (
  increase(dispatcharr_qoe_output_profile_writer_skipped_chunks_total[5m])
)

# Source and per-output-profile discontinuity boundaries
dispatcharr_qoe_input_buffer_discontinuities
dispatcharr_qoe_output_profile_buffer_discontinuities

# Locate an upstream or per-profile output gap
dispatcharr_qoe_input_buffer_last_chunk_age_seconds
dispatcharr_qoe_output_profile_buffer_last_chunk_age_seconds

# Input stopped receiving chunks while Dispatcharr still reports the channel active
dispatcharr_qoe_channel_active == 1
  and on (channel_uuid, channel_number)
dispatcharr_qoe_input_buffer_fresh == 0

# An active output profile stopped writing chunks
dispatcharr_qoe_output_profile_process_active == 1
  and on (channel_uuid, output_profile_id)
dispatcharr_qoe_output_profile_buffer_fresh == 0

# TimeControl reports dropped frames if supported by its FFmpeg build
dispatcharr_qoe_ffmpeg_progress_dropped_frames_total

# Server delivery gaps after playback has started, per output format/profile
sum by (channel_uuid, output_format, output_profile_id) (
  increase(dispatcharr_qoe_delivery_starvation_episodes_total{phase="playback"}[5m])
)
sum by (channel_uuid, output_format, output_profile_id) (
  increase(dispatcharr_qoe_delivery_starvation_seconds_total{phase="playback"}[5m])
)

# Missing chunks and HLS segments still advertised but unavailable
increase(dispatcharr_qoe_delivery_skipped_chunks_total[5m])
increase(dispatcharr_qoe_hls_events_total{event="segment_advertised_missing"}[5m])
increase(dispatcharr_qoe_hls_segment_late_total[5m])
```

If the input buffer becomes stale first, investigate the source/ingress path. If
input stays fresh while one output profile becomes stale, investigate that
profile's transcode and output buffer. If both remain fresh during a reported
rebuffer, the interruption may be downstream of Dispatcharr or inside the
player; these server-side signals cannot prove what the client decoded.

## Install in Dispatcharr

The source of truth is this `qoe/iptv_qoe_exporter` directory. The plugin-only
mode uses existing Redis metadata/events, TimeControl progress files and
buffer timestamp indexes. The new delivery/FFmpeg diagnostics require a tested
image built from this fork. Build from the repository root with
`docker/Dockerfile` and preserve `REPO_OWNER`, `REPO_NAME` and `BASE_TAG=base`.

1. Copy this directory to the production data volume as
   `/srv/docker/databases/dispatcharr/plugins/iptv_qoe_exporter/`.
2. Map port 9193 only to the Prometheus/NetBird network during a planned
   Dispatcharr restart.
3. Reload plugins in the Dispatcharr UI, enable **Dispatcharr QoE Metrics**,
   review its `persistent_service`, `network_listener` and `proxy_internals`
   capabilities, and restart Dispatcharr if the UI requests it. The listener
   capability requires a restart in current Dispatcharr plugin support.
4. Add the scrape job from `prometheus-scrape.yml` to the Prometheus config and
   reload Prometheus.
5. Confirm `up{job="dispatcharr-qoe"} == 1`, then check that
   `dispatcharr_qoe_redis_up == 1` and `dispatcharr_qoe_database_up == 1`.

## FFmpeg exporter comparison

[`domcyrus/ffmpeg_exporter`](https://github.com/domcyrus/ffmpeg_exporter) is a
useful reference and could be useful as a controlled independent probe. Its
README describes an FFprobe-based exporter started with an explicit `--input`
URL; it reports FPS/bitrate and packet/codec error signals. It does not attach
to Dispatcharr's existing FFmpeg processes. Running it against a production
feed would open another stream connection, so this plugin instead reads the
stats/progress Dispatcharr already produces. A separate probe is best trialed
on a test feed or staging channel.

## Known limitation

The optional stderr counters only see diagnostics the running FFmpeg process
actually emits. The active input profile uses `-err_detect ignore_err` and
`+discardcorrupt`: FFmpeg can continue after a decoding error, so a healthy
1.00× speed does not prove that every packet/frame was clean, and absence of a
diagnostic does not prove that the stream was clean. The patched image keeps
diagnostic counters through a fast process/manager restart in Redis; they can
still reset after the diagnostics key expires or Redis loses state. Prometheus
`increase()` handles such resets for event analysis. Without the optional patched image, the plugin-only
buffer/event/progress metrics remain available, while the stderr diagnostic
series report as unavailable.

The plugin still cannot observe rebuffer duration, decoder drops or buffer
depth inside a spectator's player. FFmpeg's dropped-frame counter describes
the server-side process, not the player's decoder. Confirming a viewer's actual
rebuffer requires client-side QoE telemetry or a synthetic player in the same
network/device class.
