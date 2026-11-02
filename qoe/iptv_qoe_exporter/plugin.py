"""Dispatcharr QoE metrics exporter.

This plugin intentionally exports aggregate channel/profile information only.
It never exports stream URLs, client IDs, usernames, IP addresses, user agents,
provider names, EPG titles, or raw FFmpeg command arguments.
"""

import logging
import math
import re
import sqlite3
import time
import uuid
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


EVENT_ACTION = "record_dispatcharr_event"
EVENTS = {
    "channel_start",
    "channel_stop",
    "channel_buffering",
    "channel_failover",
    "channel_reconnect",
    "channel_error",
    "stream_switch",
    "client_connect",
    "client_disconnect",
}
REASONS = {
    "buffering_timeout",
    "max_retries_exceeded",
    "health_monitor",
    "manual",
    "unknown",
    "none",
    "other",
}
CHANNEL_STATES = {
    "initializing",
    "connecting",
    "waiting_for_clients",
    "active",
    "buffering",
    "error",
    "stopping",
    "stopped",
    "idle",
    "unknown",
}
ACTIVE_STATES = {"initializing", "connecting", "waiting_for_clients", "active", "buffering"}
TIME_CONTROL_PROFILE = "OpenStream TimeControl"
PROGRESS_DIR = Path("/tmp")
PROGRESS_FILE_RE = re.compile(r"^omts_(.+)\.prog$")
OUTPUT_STATE_RE = re.compile(r":p(\d+):state$")
OUTPUT_DIAGNOSTICS_RE = re.compile(r"^live:channel:([^:]+):output:mpegts:p(\d+):diagnostics$")
OUTPUT_DELIVERY_RE = re.compile(
    r"^live:channel:([^:]+):output:((?:mpegts|hls|fmp4)(?::p\d+)?):delivery:diagnostics$"
)
FFMPEG_ERROR_TYPES = (
    "continuity_error",
    "corrupt_packet",
    "packet_drop",
    "decode_error",
    "timestamp_error",
    "input_error",
    "mux_error",
    "io_error",
    "other_error",
)
OUTPUT_RESTART_REASONS = ("input_discontinuity", "pmt_layout_change", "other")
OUTPUT_STOP_REASONS = ("process_exit", "stdout_eof", "reader_error")


def _text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value) if value is not None else ""


def _float(value):
    try:
        parsed = float(_text(value).strip())
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _channel_uuid(value):
    try:
        return str(uuid.UUID(_text(value)))
    except (ValueError, AttributeError, TypeError):
        return None


def _label(value):
    value = _text(value)[:160]
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _sample(name, labels, value):
    label_text = ""
    if labels:
        label_text = "{" + ",".join(
            f'{key}="{_label(val)}"' for key, val in sorted(labels.items())
        ) + "}"
    number = float(value)
    if not math.isfinite(number):
        number = 0.0
    return f"{name}{label_text} {number:.12g}"


def _metric(lines, name, kind, help_text, samples):
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} {kind}")
    lines.extend(samples)


def _normalized_reason(payload):
    raw = payload.get("reason") or payload.get("error_type")
    if not raw:
        return "none"
    value = _text(raw).strip().lower()
    return value if value in REASONS - {"none", "other"} else "other"


def _read_progress_file(path, now, max_age):
    """Read only the latest bounded FFmpeg -progress block from a TimeControl file."""
    try:
        stat = path.stat()
        age = max(0.0, now - stat.st_mtime)
        if age > max_age or stat.st_size <= 0:
            return None
        with path.open("rb") as handle:
            handle.seek(max(0, stat.st_size - 16384))
            tail = handle.read(16384).decode("utf-8", errors="replace")
        fields = {}
        for line in tail.splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key in {
                "out_time_us", "out_time_ms", "speed", "progress", "frame",
                "fps", "drop_frames", "dup_frames", "bitrate",
            }:
                fields[key] = value.strip()
        speed_text = fields.get("speed", "").removesuffix("x")
        speed = _float(speed_text)
        out_time_us = _float(fields.get("out_time_us"))
        if out_time_us is None:
            out_time_us = _float(fields.get("out_time_ms"))
        return {
            "updated": stat.st_mtime,
            "speed": speed,
            "out_time_seconds": out_time_us / 1_000_000 if out_time_us is not None else None,
            "frame": _float(fields.get("frame")),
            "fps": _float(fields.get("fps")),
            "drop_frames": _float(fields.get("drop_frames")),
            "dup_frames": _float(fields.get("dup_frames")),
            "completed": 1 if fields.get("progress") == "end" else 0,
        }
    except (OSError, ValueError):
        return None


class _MetricsHTTPServer(HTTPServer):
    allow_reuse_address = True


class Plugin:
    name = "Dispatcharr QoE Metrics"
    version = "0.3.0"
    description = "Exports channel, FFmpeg and Dispatcharr event metrics for Prometheus."
    author = "AceStream Project"

    fields = [
        {"id": "bind_host", "label": "Metrics bind address", "type": "string", "default": "0.0.0.0"},
        {"id": "port", "label": "Metrics port", "type": "number", "default": 9193},
        {"id": "stats_stale_after_seconds", "label": "FFmpeg stats freshness (seconds)", "type": "number", "default": 15},
        {"id": "progress_stale_after_seconds", "label": "FFmpeg progress freshness (seconds)", "type": "number", "default": 15},
        {"id": "buffer_stale_after_seconds", "label": "Buffer freshness threshold (seconds)", "type": "number", "default": 10},
    ]
    actions = [
        {
            "id": EVENT_ACTION,
            "label": "Dispatcharr event collector",
            "description": "Stores selected Dispatcharr events. The button shows the durable counter summary.",
            "button_label": "View counters",
            "button_variant": "light",
            "button_color": "blue",
            "events": sorted(EVENTS),
        }
    ]

    def __init__(self):
        self._settings = {}
        self._data_dir = None
        self._channel_cache = {}
        self._channel_pk_to_uuid = {}
        self._channel_index_updated = 0.0
        self._collector_errors = 0
        self._logger = logging.getLogger(__name__)

    def on_leader_acquired(self, context):
        self._settings = dict(context.get("settings") or {})
        self._data_dir = Path(context["data_dir"])
        context["start_service"](self._run_service, name="prometheus-metrics")

    def on_leader_lost(self, context):
        # Dispatcharr owns and joins the service thread before calling this hook.
        return None

    def run(self, action, params, context):
        if action != EVENT_ACTION:
            return {"status": "error", "message": "Unknown action."}

        event = _text((params or {}).get("event"))
        payload = (params or {}).get("payload")
        if event in EVENTS and isinstance(payload, dict):
            channel = _channel_uuid(payload.get("channel_id") or payload.get("channel_uuid"))
            channel = channel or "unknown"
            reason = _normalized_reason(payload)
            duration = _float(payload.get("duration")) if event == "channel_failover" else None
            duration = duration if duration is not None and 0 <= duration <= 3600 else 0.0
            event_time = time.time()
            db_path = self._event_db_path(context["data_dir"])
            with sqlite3.connect(str(db_path), timeout=5) as connection:
                connection.execute(
                    """CREATE TABLE IF NOT EXISTS event_counters (
                        event TEXT NOT NULL,
                        channel_uuid TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        count INTEGER NOT NULL DEFAULT 0,
                        last_received_timestamp REAL NOT NULL DEFAULT 0,
                        duration_seconds_total REAL NOT NULL DEFAULT 0,
                        PRIMARY KEY (event, channel_uuid, reason)
                    )"""
                )
                connection.execute(
                    """INSERT INTO event_counters
                        (event, channel_uuid, reason, count, last_received_timestamp, duration_seconds_total)
                        VALUES (?, ?, ?, 1, ?, ?)
                        ON CONFLICT(event, channel_uuid, reason) DO UPDATE SET
                            count = count + 1,
                            last_received_timestamp = excluded.last_received_timestamp,
                            duration_seconds_total = duration_seconds_total + excluded.duration_seconds_total
                    """,
                    (event, channel, reason, event_time, duration),
                )
            return {"status": "ok", "recorded": True}

        return self._counter_summary(context.get("data_dir"))

    @staticmethod
    def _event_db_path(data_dir):
        directory = Path(data_dir)
        directory.mkdir(parents=True, exist_ok=True)
        return directory / "dispatcharr-qoe-events.sqlite3"

    def _counter_summary(self, data_dir):
        try:
            path = Path(data_dir) / "dispatcharr-qoe-events.sqlite3"
            if not path.exists():
                return {"status": "ok", "counter_series": 0, "events_recorded": 0}
            with sqlite3.connect(str(path), timeout=5) as connection:
                table = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='event_counters'"
                ).fetchone()
                if not table:
                    return {"status": "ok", "counter_series": 0, "events_recorded": 0}
                row = connection.execute(
                    "SELECT COUNT(*), COALESCE(SUM(count), 0), COALESCE(MAX(last_received_timestamp), 0) FROM event_counters"
                ).fetchone()
            return {
                "status": "ok",
                "counter_series": int(row[0]),
                "events_recorded": int(row[1]),
                "last_event_received_timestamp_seconds": float(row[2]),
            }
        except sqlite3.Error:
            return {"status": "ok", "counter_series": 0, "events_recorded": 0}

    def _run_service(self, stop_event):
        host = _text(self._settings.get("bind_host", "0.0.0.0")).strip() or "0.0.0.0"
        try:
            port = int(self._settings.get("port", 9193))
        except (TypeError, ValueError):
            port = 9193
        if not 1024 <= port <= 65535:
            port = 9193

        plugin = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path == "/healthz":
                    body = b"ok\n"
                    status = 200
                    content_type = "text/plain; charset=utf-8"
                elif path == "/metrics":
                    body = plugin._render_metrics().encode("utf-8")
                    status = 200
                    content_type = "text/plain; version=0.0.4; charset=utf-8"
                else:
                    body = b"not found\n"
                    status = 404
                    content_type = "text/plain; charset=utf-8"
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt, *args):
                # Avoid logging request paths or remote addresses unnecessarily.
                return

        while not stop_event.is_set():
            server = None
            try:
                server = _MetricsHTTPServer((host, port), Handler)
                server.timeout = 1.0
                self._logger.info("Dispatcharr QoE exporter listening on %s:%s", host, port)
                while not stop_event.is_set():
                    server.handle_request()
            except OSError as exc:
                self._logger.warning("QoE exporter bind/serve error (%s); retrying", type(exc).__name__)
                stop_event.wait(5.0)
            finally:
                if server is not None:
                    try:
                        server.server_close()
                    except OSError:
                        pass

    def _load_channel_index(self, now):
        if now - self._channel_index_updated < 60.0:
            return self._channel_cache, True
        try:
            from django.db import close_old_connections
            from apps.channels.models import Channel

            close_old_connections()
            try:
                rows = list(Channel.objects.values_list("id", "uuid", "channel_number"))
                index = {str(channel_uuid): number for _pk, channel_uuid, number in rows}
                pk_to_uuid = {str(pk): str(channel_uuid) for pk, channel_uuid, _number in rows}
            finally:
                close_old_connections()
            self._channel_cache = index
            self._channel_pk_to_uuid = pk_to_uuid
            self._channel_index_updated = now
            return index, True
        except Exception as exc:
            self._collector_errors += 1
            self._logger.warning("Could not read channel index (%s)", type(exc).__name__)
            return self._channel_cache, False

    @staticmethod
    def _redis_channel_metadata(redis_client):
        from apps.proxy.live_proxy.redis_keys import RedisKeys

        channels = {}
        for key in redis_client.scan_iter(match="live:channel:*:metadata", count=200):
            key_text = _text(key)
            parts = key_text.split(":")
            if len(parts) != 4 or parts[0:2] != ["live", "channel"] or parts[3] != "metadata":
                continue
            channel = _channel_uuid(parts[2])
            if not channel:
                continue
            raw = redis_client.hgetall(key)
            metadata = {_text(k): _text(v) for k, v in (raw or {}).items()}
            channels[channel] = metadata
        return channels

    @staticmethod
    def _client_profile_stats(redis_client, channel_uuid):
        from apps.proxy.live_proxy.redis_keys import RedisKeys

        client_ids = redis_client.smembers(RedisKeys.clients(channel_uuid)) or set()
        profile_counts = defaultdict(int)
        profile_egress = defaultdict(float)
        format_counts = defaultdict(int)
        if not client_ids:
            return 0, profile_counts, profile_egress, format_counts

        pipeline = redis_client.pipeline(transaction=False)
        for client_id in client_ids:
            pipeline.hmget(
                RedisKeys.client_metadata(channel_uuid, _text(client_id)),
                "output_profile_id",
                "current_rate_KBps",
                "stats_updated_at",
                "output_format",
            )
        for result in pipeline.execute():
            profile_raw = _text(result[0]).strip() if result and result[0] is not None else "unknown"
            profile = profile_raw if profile_raw.isdigit() else "unknown"
            rate_kbps = _float(result[1]) if result and len(result) > 1 else None
            stats_updated = _float(result[2]) if result and len(result) > 2 else None
            output_format = _text(result[3]).strip() if result and len(result) > 3 else ""
            if output_format not in ("mpegts", "hls", "fmp4"):
                output_format = "unknown"
            profile_counts[profile] += 1
            format_counts[output_format] += 1
            rate_is_fresh = bool(stats_updated and 0 <= time.time() - stats_updated <= 30)
            if rate_is_fresh and rate_kbps is not None and rate_kbps > 0:
                # Dispatcharr stores this rate as bytes / 1024 / second.
                profile_egress[profile] += rate_kbps * 1024.0
        return len(client_ids), profile_counts, profile_egress, format_counts

    @staticmethod
    def _output_profile_states(redis_client, channel_uuid):
        from apps.proxy.live_proxy.redis_keys import RedisKeys

        states = {}
        pattern = f"live:channel:{channel_uuid}:output:mpegts:p*:state"
        for key in redis_client.scan_iter(match=pattern, count=100):
            key_text = _text(key)
            match = OUTPUT_STATE_RE.search(key_text)
            if not match:
                continue
            profile = match.group(1)
            value = _text(redis_client.get(key)).strip().lower()
            states[profile] = value if value in {"active", "stopped"} else "unknown"
        return states

    @staticmethod
    def _output_ffmpeg_diagnostics(redis_client, channel_uuid):
        from apps.proxy.live_proxy.redis_keys import RedisKeys

        diagnostics = {}
        pattern = f"live:channel:{channel_uuid}:output:mpegts:p*:diagnostics"
        for key in redis_client.scan_iter(match=pattern, count=100):
            match = OUTPUT_DIAGNOSTICS_RE.match(_text(key))
            if not match or _channel_uuid(match.group(1)) != channel_uuid:
                continue
            profile = match.group(2)
            raw = redis_client.hgetall(RedisKeys.output_ffmpeg_diagnostics(channel_uuid, profile))
            diagnostics[profile] = {_text(k): _text(v) for k, v in (raw or {}).items()}
        return diagnostics

    @staticmethod
    def _output_delivery_diagnostics(redis_client, channel_uuid):
        from apps.proxy.live_proxy.redis_keys import RedisKeys

        diagnostics = {}
        pattern = f"live:channel:{channel_uuid}:output:*:delivery:diagnostics"
        for key in redis_client.scan_iter(match=pattern, count=100):
            match = OUTPUT_DELIVERY_RE.match(_text(key))
            if not match or _channel_uuid(match.group(1)) != channel_uuid:
                continue
            fmt = match.group(2)
            raw = redis_client.hgetall(RedisKeys.output_delivery_diagnostics(channel_uuid, fmt))
            diagnostics[fmt] = {_text(k): _text(v) for k, v in (raw or {}).items()}
        return diagnostics

    @staticmethod
    def _buffer_summary(redis_client, key, now):
        """Summarize existing Redis chunk timestamps without reading media data."""
        pipeline = redis_client.pipeline(transaction=False)
        pipeline.zcard(key)
        pipeline.zrange(key, 0, 0, withscores=True)
        pipeline.zrange(key, -1, -1, withscores=True)
        count, oldest_rows, newest_rows = pipeline.execute()
        oldest = float(oldest_rows[0][1]) if oldest_rows else None
        newest = float(newest_rows[0][1]) if newest_rows else None
        return {
            "chunks": int(count or 0),
            "oldest_timestamp": oldest,
            "latest_timestamp": newest,
            "latest_age": max(0.0, now - newest) if newest is not None else None,
            "time_span": max(0.0, newest - oldest) if newest is not None and oldest is not None else 0.0,
        }

    def _append_event_metrics(self, lines):
        name = "dispatcharr_qoe_dispatcharr_events_total"
        duration_name = "dispatcharr_qoe_dispatcharr_event_duration_seconds_total"
        last_name = "dispatcharr_qoe_last_dispatcharr_event_received_timestamp_seconds"
        event_samples, duration_samples, last_samples = [], [], []
        try:
            path = Path(self._data_dir) / "dispatcharr-qoe-events.sqlite3"
            rows = []
            if path.exists():
                with sqlite3.connect(str(path), timeout=2) as connection:
                    table = connection.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='event_counters'"
                    ).fetchone()
                    if table:
                        rows = connection.execute(
                            "SELECT event, channel_uuid, reason, count, last_received_timestamp, duration_seconds_total "
                            "FROM event_counters ORDER BY event, channel_uuid, reason"
                        ).fetchall()
            for event, channel, reason, count, timestamp, duration in rows:
                labels = {"channel_uuid": channel, "event": event, "reason": reason}
                event_samples.append(_sample(name, labels, count))
                duration_samples.append(_sample(duration_name, labels, duration))
                last_samples.append(_sample(last_name, labels, timestamp))
        except sqlite3.Error as exc:
            self._collector_errors += 1
            self._logger.warning("Could not read event counter store (%s)", type(exc).__name__)
        _metric(lines, name, "counter", "Dispatcharr events delivered to the plugin event hook.", event_samples)
        _metric(lines, duration_name, "counter", "Total server-reported duration carried by channel_failover events, in seconds.", duration_samples)
        _metric(lines, last_name, "gauge", "Unix time when the plugin received the latest event callback.", last_samples)

    def _append_progress_metrics(self, lines, known_channel_ids, metadata_by_channel, now, channel_pk_to_uuid=None):
        present_name = "dispatcharr_qoe_ffmpeg_progress_present"
        speed_name = "dispatcharr_qoe_ffmpeg_progress_speed_ratio"
        out_time_name = "dispatcharr_qoe_ffmpeg_progress_out_time_seconds"
        updated_name = "dispatcharr_qoe_ffmpeg_progress_updated_timestamp_seconds"
        completed_name = "dispatcharr_qoe_ffmpeg_progress_completed"
        frames_name = "dispatcharr_qoe_ffmpeg_progress_frames"
        fps_name = "dispatcharr_qoe_ffmpeg_progress_fps"
        dropped_name = "dispatcharr_qoe_ffmpeg_progress_dropped_frames_total"
        duplicated_name = "dispatcharr_qoe_ffmpeg_progress_duplicated_frames_total"
        max_age = _float(self._settings.get("progress_stale_after_seconds", 15)) or 15.0
        values = {}

        for channel, metadata in metadata_by_channel.items():
            if metadata.get("stream_profile") == TIME_CONTROL_PROFILE:
                values[channel] = None

        for path in PROGRESS_DIR.glob("omts_*.prog"):
            match = PROGRESS_FILE_RE.match(path.name)
            file_channel = match.group(1) if match else ""
            channel = _channel_uuid(file_channel) or (channel_pk_to_uuid or {}).get(file_channel)
            if not channel or channel not in known_channel_ids:
                continue
            progress = _read_progress_file(path, now, max_age)
            if progress is not None:
                values[channel] = progress

        present_samples, speed_samples, out_time_samples, updated_samples, completed_samples = [], [], [], [], []
        frames_samples, fps_samples, dropped_samples, duplicated_samples = [], [], [], []
        for channel, progress in sorted(values.items()):
            labels = {"channel_uuid": channel}
            present_samples.append(_sample(present_name, labels, 1 if progress is not None else 0))
            if progress is None:
                continue
            speed_samples.append(_sample(speed_name, labels, progress["speed"] or 0.0))
            if progress["out_time_seconds"] is not None:
                out_time_samples.append(_sample(out_time_name, labels, progress["out_time_seconds"]))
            updated_samples.append(_sample(updated_name, labels, progress["updated"]))
            completed_samples.append(_sample(completed_name, labels, progress["completed"]))
            if progress["frame"] is not None:
                frames_samples.append(_sample(frames_name, labels, progress["frame"]))
            if progress["fps"] is not None:
                fps_samples.append(_sample(fps_name, labels, progress["fps"]))
            if progress["drop_frames"] is not None:
                dropped_samples.append(_sample(dropped_name, labels, progress["drop_frames"]))
            if progress["dup_frames"] is not None:
                duplicated_samples.append(_sample(duplicated_name, labels, progress["dup_frames"]))

        _metric(lines, present_name, "gauge", "Whether a fresh OpenStream TimeControl FFmpeg progress file is available.", present_samples)
        _metric(lines, speed_name, "gauge", "FFmpeg progress speed ratio for the TimeControl input process.", speed_samples)
        _metric(lines, out_time_name, "gauge", "Raw FFmpeg output timeline in seconds; TimeControl applies output_ts_offset, so this is not wall-clock latency.", out_time_samples)
        _metric(lines, updated_name, "gauge", "Unix timestamp of the latest FFmpeg TimeControl progress file update.", updated_samples)
        _metric(lines, completed_name, "gauge", "Whether the latest progress block says progress=end.", completed_samples)
        _metric(lines, frames_name, "gauge", "Latest input FFmpeg progress frame number.", frames_samples)
        _metric(lines, fps_name, "gauge", "Input FFmpeg progress frames per second.", fps_samples)
        _metric(lines, dropped_name, "counter", "Cumulative input FFmpeg dropped-frame count for the current process session.", dropped_samples)
        _metric(lines, duplicated_name, "counter", "Cumulative input FFmpeg duplicated-frame count for the current process session.", duplicated_samples)

    def _render_metrics(self):
        started = time.monotonic()
        now = time.time()
        lines = []
        redis_up = 0
        database_up = 0
        redis_client = None
        try:
            from core.utils import RedisClient

            redis_client = RedisClient.get_client(
                max_retries=1,
                retry_interval=0,
                disable_persistence=False,
            )
            redis_up = 1 if redis_client is not None and redis_client.ping() else 0
        except Exception as exc:
            self._collector_errors += 1
            self._logger.warning("Redis metrics collection failed (%s)", type(exc).__name__)

        channels, database_up = self._load_channel_index(now)
        metadata_by_channel = {}
        if redis_up:
            try:
                metadata_by_channel = self._redis_channel_metadata(redis_client)
            except Exception as exc:
                redis_up = 0
                self._collector_errors += 1
                self._logger.warning("Redis channel snapshot failed (%s)", type(exc).__name__)

        _metric(lines, "dispatcharr_qoe_exporter_up", "gauge", "Whether the custom Dispatcharr metrics exporter is serving.", ["dispatcharr_qoe_exporter_up 1"])
        _metric(lines, "dispatcharr_qoe_redis_up", "gauge", "Whether Redis was reachable during this collection.", [_sample("dispatcharr_qoe_redis_up", {}, redis_up)])
        _metric(lines, "dispatcharr_qoe_database_up", "gauge", "Whether the channel index is available; the database index is refreshed every 60 seconds.", [_sample("dispatcharr_qoe_database_up", {}, database_up)])

        if redis_up:
            from apps.proxy.live_proxy.redis_keys import RedisKeys

            all_channel_ids = set(channels) | set(metadata_by_channel)
            number_labels = {
                channel: (f"{float(number):.1f}" if number is not None else "")
                for channel, number in channels.items()
            }
            metric_samples = defaultdict(list)
            stats_max_age = _float(self._settings.get("stats_stale_after_seconds", 15)) or 15.0
            buffer_stale_after = _float(self._settings.get("buffer_stale_after_seconds", 10)) or 10.0

            for channel in sorted(all_channel_ids):
                metadata = metadata_by_channel.get(channel, {})
                raw_state = metadata.get("state", "idle").strip().lower() or "idle"
                state = raw_state if raw_state in CHANNEL_STATES else "unknown"
                stream_profile = metadata.get("stream_profile", "none").strip() or "none"
                channel_labels = {
                    "channel_uuid": channel,
                    "channel_number": number_labels.get(channel, ""),
                }
                metadata_present = 1 if metadata else 0
                active = 1 if metadata_present and state in ACTIVE_STATES else 0
                metric_samples["dispatcharr_qoe_channel_metadata_present"].append(
                    _sample("dispatcharr_qoe_channel_metadata_present", channel_labels, metadata_present)
                )
                metric_samples["dispatcharr_qoe_channel_active"].append(
                    _sample("dispatcharr_qoe_channel_active", channel_labels, active)
                )
                metric_samples["dispatcharr_qoe_channel_state"].append(
                    _sample("dispatcharr_qoe_channel_state", {**channel_labels, "state": state}, 1)
                )
                metric_samples["dispatcharr_qoe_channel_stream_profile_info"].append(
                    _sample("dispatcharr_qoe_channel_stream_profile_info", {**channel_labels, "input_profile": stream_profile}, 1)
                )

                client_count = 0
                profile_counts = defaultdict(int)
                profile_egress = defaultdict(float)
                format_counts = defaultdict(int)
                if metadata_present:
                    try:
                        client_count, profile_counts, profile_egress, format_counts = self._client_profile_stats(redis_client, channel)
                    except Exception as exc:
                        self._collector_errors += 1
                        self._logger.warning("Could not read client aggregate (%s)", type(exc).__name__)
                metric_samples["dispatcharr_qoe_channel_clients"].append(
                    _sample("dispatcharr_qoe_channel_clients", channel_labels, client_count)
                )
                for output_format, count in sorted(format_counts.items()):
                    metric_samples["dispatcharr_qoe_output_format_clients"].append(
                        _sample("dispatcharr_qoe_output_format_clients", {"channel_uuid": channel, "output_format": output_format}, count)
                    )

                try:
                    input_buffer = self._buffer_summary(
                        redis_client,
                        RedisKeys.chunk_timestamps(channel),
                        now,
                    )
                except Exception as exc:
                    input_buffer = {"chunks": 0, "latest_timestamp": None, "latest_age": None, "time_span": 0.0}
                    self._collector_errors += 1
                    self._logger.warning("Could not read input buffer timestamps (%s)", type(exc).__name__)
                metric_samples["dispatcharr_qoe_input_buffer_chunks_retained"].append(
                    _sample("dispatcharr_qoe_input_buffer_chunks_retained", channel_labels, input_buffer["chunks"])
                )
                metric_samples["dispatcharr_qoe_input_buffer_time_span_seconds_estimate"].append(
                    _sample("dispatcharr_qoe_input_buffer_time_span_seconds_estimate", channel_labels, input_buffer["time_span"])
                )
                input_buffer_fresh = bool(
                    input_buffer["latest_age"] is not None
                    and input_buffer["latest_age"] <= buffer_stale_after
                )
                metric_samples["dispatcharr_qoe_input_buffer_fresh"].append(
                    _sample("dispatcharr_qoe_input_buffer_fresh", channel_labels, 1 if input_buffer_fresh else 0)
                )
                if input_buffer["latest_age"] is not None:
                    metric_samples["dispatcharr_qoe_input_buffer_last_chunk_age_seconds"].append(
                        _sample("dispatcharr_qoe_input_buffer_last_chunk_age_seconds", channel_labels, input_buffer["latest_age"])
                    )
                    metric_samples["dispatcharr_qoe_input_buffer_last_chunk_timestamp_seconds"].append(
                        _sample("dispatcharr_qoe_input_buffer_last_chunk_timestamp_seconds", channel_labels, input_buffer["latest_timestamp"])
                    )

                # Keep FFmpeg diagnostics in a dedicated Redis hash. Error labels
                # are a fixed allowlist and no raw stderr text is exported.
                input_diag_key = RedisKeys.input_ffmpeg_diagnostics(channel)
                try:
                    input_diag_present = bool(redis_client.exists(input_diag_key))
                    input_diag_raw = redis_client.hgetall(input_diag_key) if input_diag_present else {}
                except Exception as exc:
                    input_diag_present = False
                    input_diag_raw = {}
                    self._collector_errors += 1
                    self._logger.warning("Could not read input FFmpeg diagnostics (%s)", type(exc).__name__)
                input_diag = {_text(k): _text(v) for k, v in (input_diag_raw or {}).items()}
                input_diag_labels = {
                    "channel_uuid": channel,
                    "scope": "input",
                    "output_profile_id": "none",
                }
                metric_samples["dispatcharr_qoe_ffmpeg_diagnostics_present"].append(
                    _sample("dispatcharr_qoe_ffmpeg_diagnostics_present", input_diag_labels, 1 if input_diag_present else 0)
                )
                input_stats_updated = _float(input_diag.get("ffmpeg_stats_updated_timestamp")) or 0.0
                input_diag_stats_fresh = bool(
                    input_stats_updated and 0 <= now - input_stats_updated <= stats_max_age
                )
                metric_samples["dispatcharr_qoe_ffmpeg_stats_present"].append(
                    _sample("dispatcharr_qoe_ffmpeg_stats_present", input_diag_labels, 1 if input_diag_stats_fresh else 0)
                )
                if input_diag_present:
                    for error_type in FFMPEG_ERROR_TYPES:
                        error_labels = {**input_diag_labels, "error_type": error_type}
                        metric_samples["dispatcharr_qoe_ffmpeg_errors_total"].append(
                            _sample("dispatcharr_qoe_ffmpeg_errors_total", error_labels, _float(input_diag.get(f"ffmpeg_error_{error_type}_total")) or 0)
                        )
                    last_error_type = input_diag.get("ffmpeg_last_error_type", "none")
                    last_error_time = _float(input_diag.get("ffmpeg_last_error_timestamp")) or 0.0
                    if last_error_type in FFMPEG_ERROR_TYPES and last_error_time > 0:
                        metric_samples["dispatcharr_qoe_ffmpeg_last_error_timestamp_seconds"].append(
                            _sample(
                                "dispatcharr_qoe_ffmpeg_last_error_timestamp_seconds",
                                {**input_diag_labels, "error_type": last_error_type},
                                last_error_time,
                            )
                        )
                try:
                    input_discontinuities = redis_client.zcard(RedisKeys.buffer_discontinuities(channel))
                except Exception as exc:
                    input_discontinuities = 0
                    self._collector_errors += 1
                    self._logger.warning("Could not read input discontinuities (%s)", type(exc).__name__)
                metric_samples["dispatcharr_qoe_input_buffer_discontinuities"].append(
                    _sample("dispatcharr_qoe_input_buffer_discontinuities", channel_labels, input_discontinuities)
                )
                if input_diag_stats_fresh:
                    input_stat_values = {
                        "dispatcharr_qoe_ffmpeg_stats_updated_timestamp_seconds": input_stats_updated,
                        "dispatcharr_qoe_ffmpeg_speed_ratio": _float(input_diag.get("ffmpeg_speed")),
                        "dispatcharr_qoe_ffmpeg_fps": _float(input_diag.get("ffmpeg_fps")),
                        "dispatcharr_qoe_ffmpeg_output_bytes_per_second": (
                            (_float(input_diag.get("ffmpeg_output_bitrate_bits_per_second")) or 0.0) / 8.0
                        ),
                        "dispatcharr_qoe_ffmpeg_frames": _float(input_diag.get("ffmpeg_frame")),
                    }
                    for name, value in input_stat_values.items():
                        if value is not None:
                            metric_samples[name].append(_sample(name, input_diag_labels, value))

                stats_updated = _float(metadata.get("ffmpeg_stats_updated")) or 0.0
                stats_fresh = bool(stats_updated and 0 <= now - stats_updated <= stats_max_age)
                metric_samples["dispatcharr_qoe_channel_ffmpeg_stats_present"].append(
                    _sample("dispatcharr_qoe_channel_ffmpeg_stats_present", channel_labels, 1 if stats_fresh else 0)
                )
                metric_samples["dispatcharr_qoe_channel_ffmpeg_stats_updated_timestamp_seconds"].append(
                    _sample("dispatcharr_qoe_channel_ffmpeg_stats_updated_timestamp_seconds", channel_labels, stats_updated if metadata_present else 0)
                )
                ffmpeg_values = {
                    "dispatcharr_qoe_channel_ffmpeg_speed_ratio": (_float(metadata.get("ffmpeg_speed")) if stats_fresh else None),
                    "dispatcharr_qoe_channel_ffmpeg_fps": (_float(metadata.get("ffmpeg_fps")) if stats_fresh else None),
                    "dispatcharr_qoe_channel_actual_fps": (_float(metadata.get("actual_fps")) if stats_fresh else None),
                    "dispatcharr_qoe_channel_ffmpeg_output_bytes_per_second": (
                        (_float(metadata.get("ffmpeg_output_bitrate")) * 1000 / 8)
                        if stats_fresh and _float(metadata.get("ffmpeg_output_bitrate")) is not None
                        else None
                    ),
                    "dispatcharr_qoe_channel_input_bytes": (_float(metadata.get("total_bytes")) if metadata_present else None),
                }
                for name, value in ffmpeg_values.items():
                    metric_samples[name].append(_sample(name, channel_labels, value or 0.0))

                # Aggregate by output profile, never by client/session identifiers.
                states = {}
                if metadata_present:
                    try:
                        states = self._output_profile_states(redis_client, channel)
                    except Exception as exc:
                        self._collector_errors += 1
                        self._logger.warning("Could not read output profile state (%s)", type(exc).__name__)
                try:
                    output_diagnostics = self._output_ffmpeg_diagnostics(redis_client, channel)
                except Exception as exc:
                    output_diagnostics = {}
                    self._collector_errors += 1
                    self._logger.warning("Could not read output FFmpeg diagnostics (%s)", type(exc).__name__)
                profile_ids = set(profile_counts) | set(states) | set(output_diagnostics)
                for profile in sorted(profile_ids):
                    output_labels = {"channel_uuid": channel, "output_profile_id": profile}
                    metric_samples["dispatcharr_qoe_output_profile_clients"].append(
                        _sample("dispatcharr_qoe_output_profile_clients", output_labels, profile_counts.get(profile, 0))
                    )
                    metric_samples["dispatcharr_qoe_output_profile_egress_bytes_per_second"].append(
                        _sample("dispatcharr_qoe_output_profile_egress_bytes_per_second", output_labels, profile_egress.get(profile, 0.0))
                    )
                    metric_samples["dispatcharr_qoe_output_profile_process_active"].append(
                        _sample("dispatcharr_qoe_output_profile_process_active", output_labels, 1 if states.get(profile) == "active" else 0)
                    )
                    try:
                        output_discontinuities = redis_client.zcard(
                            RedisKeys.output_buffer_discontinuities(channel, f"mpegts:p{profile}")
                        )
                    except Exception as exc:
                        output_discontinuities = 0
                        self._collector_errors += 1
                        self._logger.warning("Could not read output discontinuities (%s)", type(exc).__name__)
                    metric_samples["dispatcharr_qoe_output_profile_buffer_discontinuities"].append(
                        _sample(
                            "dispatcharr_qoe_output_profile_buffer_discontinuities",
                            output_labels,
                            output_discontinuities,
                        )
                    )
                    try:
                        output_buffer = self._buffer_summary(
                            redis_client,
                            RedisKeys.output_chunk_timestamps(channel, f"mpegts:p{profile}"),
                            now,
                        )
                    except Exception as exc:
                        output_buffer = {"chunks": 0, "latest_timestamp": None, "latest_age": None, "time_span": 0.0}
                        self._collector_errors += 1
                        self._logger.warning("Could not read output buffer timestamps (%s)", type(exc).__name__)
                    metric_samples["dispatcharr_qoe_output_profile_buffer_chunks_retained"].append(
                        _sample("dispatcharr_qoe_output_profile_buffer_chunks_retained", output_labels, output_buffer["chunks"])
                    )
                    metric_samples["dispatcharr_qoe_output_profile_buffer_time_span_seconds_estimate"].append(
                        _sample("dispatcharr_qoe_output_profile_buffer_time_span_seconds_estimate", output_labels, output_buffer["time_span"])
                    )
                    output_buffer_fresh = bool(
                        output_buffer["latest_age"] is not None
                        and output_buffer["latest_age"] <= buffer_stale_after
                    )
                    metric_samples["dispatcharr_qoe_output_profile_buffer_fresh"].append(
                        _sample("dispatcharr_qoe_output_profile_buffer_fresh", output_labels, 1 if output_buffer_fresh else 0)
                    )
                    if output_buffer["latest_age"] is not None:
                        metric_samples["dispatcharr_qoe_output_profile_buffer_last_chunk_age_seconds"].append(
                            _sample("dispatcharr_qoe_output_profile_buffer_last_chunk_age_seconds", output_labels, output_buffer["latest_age"])
                        )
                        metric_samples["dispatcharr_qoe_output_profile_buffer_last_chunk_timestamp_seconds"].append(
                            _sample("dispatcharr_qoe_output_profile_buffer_last_chunk_timestamp_seconds", output_labels, output_buffer["latest_timestamp"])
                        )

                    output_diag = output_diagnostics.get(profile, {})
                    output_diag_labels = {
                        "channel_uuid": channel,
                        "scope": "output",
                        "output_profile_id": profile,
                    }
                    output_diag_present = bool(output_diag)
                    metric_samples["dispatcharr_qoe_ffmpeg_diagnostics_present"].append(
                        _sample("dispatcharr_qoe_ffmpeg_diagnostics_present", output_diag_labels, 1 if output_diag_present else 0)
                    )
                    output_stats_updated = _float(output_diag.get("ffmpeg_stats_updated_timestamp")) or 0.0
                    output_stats_fresh = bool(
                        output_stats_updated and 0 <= now - output_stats_updated <= stats_max_age
                    )
                    metric_samples["dispatcharr_qoe_ffmpeg_stats_present"].append(
                        _sample("dispatcharr_qoe_ffmpeg_stats_present", output_diag_labels, 1 if output_stats_fresh else 0)
                    )
                    if output_diag_present:
                        for reason in OUTPUT_RESTART_REASONS:
                            metric_samples["dispatcharr_qoe_output_profile_ffmpeg_restarts_total"].append(
                                _sample(
                                    "dispatcharr_qoe_output_profile_ffmpeg_restarts_total",
                                    {**output_labels, "reason": reason},
                                    _float(output_diag.get(f"ffmpeg_restart_{reason}_total")) or 0,
                                )
                            )
                        for reason in OUTPUT_STOP_REASONS:
                            metric_samples["dispatcharr_qoe_output_profile_ffmpeg_unexpected_stops_total"].append(
                                _sample(
                                    "dispatcharr_qoe_output_profile_ffmpeg_unexpected_stops_total",
                                    {**output_labels, "reason": reason},
                                    _float(output_diag.get(f"ffmpeg_unexpected_stop_{reason}_total")) or 0,
                                )
                            )
                        metric_samples["dispatcharr_qoe_output_profile_writer_skipped_chunks_total"].append(
                            _sample(
                                "dispatcharr_qoe_output_profile_writer_skipped_chunks_total",
                                output_labels,
                                _float(output_diag.get("ffmpeg_writer_skipped_chunks_total")) or 0,
                            )
                        )
                        metric_samples["dispatcharr_qoe_output_profile_ffmpeg_last_stop_timestamp_seconds"].append(
                            _sample(
                                "dispatcharr_qoe_output_profile_ffmpeg_last_stop_timestamp_seconds",
                                output_labels,
                                _float(output_diag.get("ffmpeg_last_stop_timestamp")) or 0,
                            )
                        )
                        for error_type in FFMPEG_ERROR_TYPES:
                            error_labels = {**output_diag_labels, "error_type": error_type}
                            metric_samples["dispatcharr_qoe_ffmpeg_errors_total"].append(
                                _sample("dispatcharr_qoe_ffmpeg_errors_total", error_labels, _float(output_diag.get(f"ffmpeg_error_{error_type}_total")) or 0)
                            )
                        last_error_type = output_diag.get("ffmpeg_last_error_type", "none")
                        last_error_time = _float(output_diag.get("ffmpeg_last_error_timestamp")) or 0.0
                        if last_error_type in FFMPEG_ERROR_TYPES and last_error_time > 0:
                            metric_samples["dispatcharr_qoe_ffmpeg_last_error_timestamp_seconds"].append(
                                _sample(
                                    "dispatcharr_qoe_ffmpeg_last_error_timestamp_seconds",
                                    {**output_diag_labels, "error_type": last_error_type},
                                    last_error_time,
                                )
                            )
                    if output_stats_fresh:
                        output_stat_values = {
                            "dispatcharr_qoe_ffmpeg_stats_updated_timestamp_seconds": output_stats_updated,
                            "dispatcharr_qoe_ffmpeg_speed_ratio": _float(output_diag.get("ffmpeg_speed")),
                            "dispatcharr_qoe_ffmpeg_fps": _float(output_diag.get("ffmpeg_fps")),
                            "dispatcharr_qoe_ffmpeg_output_bytes_per_second": (
                                (_float(output_diag.get("ffmpeg_output_bitrate_bits_per_second")) or 0.0) / 8.0
                            ),
                            "dispatcharr_qoe_ffmpeg_frames": _float(output_diag.get("ffmpeg_frame")),
                        }
                        for name, value in output_stat_values.items():
                            if value is not None:
                                metric_samples[name].append(_sample(name, output_diag_labels, value))

                try:
                    delivery_diagnostics = self._output_delivery_diagnostics(redis_client, channel)
                except Exception as exc:
                    delivery_diagnostics = {}
                    self._collector_errors += 1
                    self._logger.warning("Could not read delivery diagnostics (%s)", type(exc).__name__)
                for fmt, delivery in sorted(delivery_diagnostics.items()):
                    parts = fmt.split(":", 1)
                    delivery_labels = {
                        "channel_uuid": channel,
                        "output_format": parts[0],
                        "output_profile_id": parts[1][1:] if len(parts) > 1 else "none",
                    }
                    for phase in ("startup", "playback"):
                        for outcome in ("recovered", "ended"):
                            metric_samples["dispatcharr_qoe_delivery_starvation_episodes_total"].append(
                                _sample(
                                    "dispatcharr_qoe_delivery_starvation_episodes_total",
                                    {**delivery_labels, "phase": phase, "outcome": outcome},
                                    _float(delivery.get(f"starvation_{phase}_{outcome}_total")) or 0,
                                )
                            )
                        metric_samples["dispatcharr_qoe_delivery_starvation_seconds_total"].append(
                            _sample(
                                "dispatcharr_qoe_delivery_starvation_seconds_total",
                                {**delivery_labels, "phase": phase},
                                _float(delivery.get(f"starvation_{phase}_seconds_total")) or 0,
                            )
                        )
                    for field, metric in (
                        ("skipped_chunks_total", "dispatcharr_qoe_delivery_skipped_chunks_total"),
                        ("keepalive_packets_total", "dispatcharr_qoe_delivery_keepalive_packets_total"),
                        ("last_starvation_timestamp", "dispatcharr_qoe_delivery_last_starvation_timestamp_seconds"),
                        ("last_starvation_seconds", "dispatcharr_qoe_delivery_last_starvation_seconds"),
                    ):
                        metric_samples[metric].append(
                            _sample(metric, delivery_labels, _float(delivery.get(field)) or 0)
                        )
                    if parts[0] == "hls":
                        for field, metric in (
                            ("hls_segments_published_total", "dispatcharr_qoe_hls_segments_published_total"),
                            ("hls_segment_late_total", "dispatcharr_qoe_hls_segment_late_total"),
                            ("hls_last_segment_timestamp", "dispatcharr_qoe_hls_last_segment_timestamp_seconds"),
                            ("hls_last_publish_gap_seconds", "dispatcharr_qoe_hls_last_publish_gap_seconds"),
                        ):
                            metric_samples[metric].append(
                                _sample(metric, delivery_labels, _float(delivery.get(field)) or 0)
                            )
                        for event in (
                            "playlist_stale", "playlist_malformed",
                            "playlist_cold_start", "playlist_cold_start_timeout",
                            "segment_expired", "segment_advertised_missing",
                        ):
                            metric_samples["dispatcharr_qoe_hls_events_total"].append(
                                _sample(
                                    "dispatcharr_qoe_hls_events_total",
                                    {**delivery_labels, "event": event},
                                    _float(delivery.get(f"hls_{event}_total")) or 0,
                                )
                            )

            specs = {
                "dispatcharr_qoe_channel_metadata_present": ("gauge", "Whether a Redis live-channel metadata hash exists for this configured channel."),
                "dispatcharr_qoe_channel_active": ("gauge", "Whether Dispatcharr reports a live channel manager state."),
                "dispatcharr_qoe_channel_state": ("gauge", "One-hot Dispatcharr channel state."),
                "dispatcharr_qoe_channel_stream_profile_info": ("gauge", "Current input stream profile reported by Dispatcharr."),
                "dispatcharr_qoe_channel_clients": ("gauge", "Number of client IDs in the Dispatcharr channel client set."),
                "dispatcharr_qoe_output_format_clients": ("gauge", "Current clients grouped by bounded output format."),
                "dispatcharr_qoe_channel_ffmpeg_stats_present": ("gauge", "Whether the latest Redis FFmpeg stats are within the configured freshness window."),
                "dispatcharr_qoe_channel_ffmpeg_stats_updated_timestamp_seconds": ("gauge", "Unix timestamp of the latest input FFmpeg stats update."),
                "dispatcharr_qoe_channel_ffmpeg_speed_ratio": ("gauge", "Input FFmpeg processing speed relative to real time."),
                "dispatcharr_qoe_channel_ffmpeg_fps": ("gauge", "Input FFmpeg reported frames per second."),
                "dispatcharr_qoe_channel_actual_fps": ("gauge", "Input FFmpeg derived source frames per second."),
                "dispatcharr_qoe_channel_ffmpeg_output_bytes_per_second": ("gauge", "Input FFmpeg output bitrate converted to bytes per second."),
                "dispatcharr_qoe_channel_input_bytes": ("gauge", "Input bytes processed in the current Redis channel session."),
                "dispatcharr_qoe_output_profile_clients": ("gauge", "Current clients grouped by output profile."),
                "dispatcharr_qoe_output_profile_egress_bytes_per_second": ("gauge", "Current client egress rate grouped by output profile."),
                "dispatcharr_qoe_output_profile_process_active": ("gauge", "Whether Redis reports an active output-profile FFmpeg process."),
                "dispatcharr_qoe_output_profile_ffmpeg_restarts_total": ("counter", "Successful output FFmpeg replacements by bounded cause."),
                "dispatcharr_qoe_output_profile_ffmpeg_unexpected_stops_total": ("counter", "Unplanned output FFmpeg reader/process stops by bounded cause."),
                "dispatcharr_qoe_output_profile_ffmpeg_last_stop_timestamp_seconds": ("gauge", "Unix timestamp of the latest unplanned output FFmpeg stop."),
                "dispatcharr_qoe_output_profile_writer_skipped_chunks_total": ("counter", "Input buffer chunks skipped by the output profile writer after falling behind."),
                "dispatcharr_qoe_delivery_starvation_episodes_total": ("counter", "Server delivery gaps of at least two seconds, by startup/playback phase and recovery outcome."),
                "dispatcharr_qoe_delivery_starvation_seconds_total": ("counter", "Total server delivery gap duration from closed episodes of at least two seconds."),
                "dispatcharr_qoe_delivery_skipped_chunks_total": ("counter", "Chunks skipped by a TS or fMP4 client after buffer expiry."),
                "dispatcharr_qoe_delivery_keepalive_packets_total": ("counter", "TS keepalive packets emitted during completed server delivery gaps."),
                "dispatcharr_qoe_delivery_last_starvation_timestamp_seconds": ("gauge", "Unix timestamp of latest completed server delivery gap."),
                "dispatcharr_qoe_delivery_last_starvation_seconds": ("gauge", "Duration of latest completed server delivery gap."),
                "dispatcharr_qoe_hls_events_total": ("counter", "HLS playlist/segment outcomes with a fixed event vocabulary."),
                "dispatcharr_qoe_hls_segments_published_total": ("counter", "HLS segments successfully written to the output buffer."),
                "dispatcharr_qoe_hls_segment_late_total": ("counter", "HLS segment publish gaps exceeding 1.5 times the advertised target duration."),
                "dispatcharr_qoe_hls_last_segment_timestamp_seconds": ("gauge", "Unix timestamp of latest published HLS segment."),
                "dispatcharr_qoe_hls_last_publish_gap_seconds": ("gauge", "Wall-clock time between the latest two HLS segment publications."),
                "dispatcharr_qoe_ffmpeg_diagnostics_present": ("gauge", "Whether Dispatcharr initialized an aggregate FFmpeg diagnostics hash for this process scope."),
                "dispatcharr_qoe_ffmpeg_errors_total": ("counter", "FFmpeg stderr diagnostics classified into a fixed category; raw lines are not stored."),
                "dispatcharr_qoe_ffmpeg_last_error_timestamp_seconds": ("gauge", "Unix timestamp of the latest classified FFmpeg stderr diagnostic."),
                "dispatcharr_qoe_ffmpeg_stats_present": ("gauge", "Whether fresh FFmpeg stderr progress stats are available."),
                "dispatcharr_qoe_ffmpeg_stats_updated_timestamp_seconds": ("gauge", "Unix timestamp of the latest parsed FFmpeg stderr progress stats."),
                "dispatcharr_qoe_ffmpeg_speed_ratio": ("gauge", "FFmpeg processing speed relative to real time, parsed from stderr progress."),
                "dispatcharr_qoe_ffmpeg_fps": ("gauge", "FFmpeg progress frames per second, parsed from stderr."),
                "dispatcharr_qoe_ffmpeg_output_bytes_per_second": ("gauge", "FFmpeg progress output bitrate converted to bytes per second."),
                "dispatcharr_qoe_ffmpeg_frames": ("gauge", "Latest FFmpeg progress frame number."),
                "dispatcharr_qoe_input_buffer_discontinuities": ("gauge", "Number of source discontinuity boundaries retained in the active input buffer."),
                "dispatcharr_qoe_input_buffer_fresh": ("gauge", "Whether an input chunk was written within the configured freshness threshold."),
                "dispatcharr_qoe_input_buffer_last_chunk_timestamp_seconds": ("gauge", "Unix timestamp of the most recent input-buffer chunk write."),
                "dispatcharr_qoe_input_buffer_last_chunk_age_seconds": ("gauge", "Age of the most recent input-buffer chunk write."),
                "dispatcharr_qoe_input_buffer_chunks_retained": ("gauge", "Number of input-buffer chunks still represented in the timestamp index."),
                "dispatcharr_qoe_input_buffer_time_span_seconds_estimate": ("gauge", "Wall-clock span between oldest and newest retained input chunk timestamps; not exact playable duration."),
                "dispatcharr_qoe_output_profile_buffer_discontinuities": ("gauge", "Number of output-profile restart/discontinuity boundaries retained in its active buffer."),
                "dispatcharr_qoe_output_profile_buffer_fresh": ("gauge", "Whether an output-profile chunk was written within the configured freshness threshold."),
                "dispatcharr_qoe_output_profile_buffer_last_chunk_timestamp_seconds": ("gauge", "Unix timestamp of the most recent output-profile chunk write."),
                "dispatcharr_qoe_output_profile_buffer_last_chunk_age_seconds": ("gauge", "Age of the most recent output-profile chunk write."),
                "dispatcharr_qoe_output_profile_buffer_chunks_retained": ("gauge", "Number of output-profile chunks still represented in the timestamp index."),
                "dispatcharr_qoe_output_profile_buffer_time_span_seconds_estimate": ("gauge", "Wall-clock span between oldest and newest retained output-profile chunk timestamps; not exact playable duration."),
            }
            for name, (kind, help_text) in specs.items():
                _metric(lines, name, kind, help_text, metric_samples[name])

            self._append_progress_metrics(
                lines,
                all_channel_ids,
                metadata_by_channel,
                now,
                self._channel_pk_to_uuid,
            )
        else:
            self._append_progress_metrics(lines, set(), {}, now, {})

        self._append_event_metrics(lines)
        _metric(
            lines,
            "dispatcharr_qoe_collector_errors_total",
            "counter",
            "Collection or event-store errors observed by this exporter process.",
            [_sample("dispatcharr_qoe_collector_errors_total", {}, self._collector_errors)],
        )
        _metric(
            lines,
            "dispatcharr_qoe_collection_duration_seconds",
            "gauge",
            "Time spent collecting and rendering the current metrics snapshot.",
            [_sample("dispatcharr_qoe_collection_duration_seconds", {}, time.monotonic() - started)],
        )
        return "\n".join(lines) + "\n"
