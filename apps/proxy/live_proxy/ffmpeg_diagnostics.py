"""Low-cardinality FFmpeg diagnostics shared by live input/output managers."""

import math
import re
import time


ERROR_TYPES = (
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

RESTART_REASONS = ("input_discontinuity", "pmt_layout_change", "other")
STOP_REASONS = ("process_exit", "stdout_eof", "reader_error")

_ERROR_PATTERNS = (
    ("continuity_error", re.compile(r"(?:continuity(?: check)? (?:failed|error)|continuity counter.{0,40}(?:not increasing|error|invalid))", re.I)),
    ("corrupt_packet", re.compile(r"(?:corrupt(?:ed)?\s+packet|packet.{0,80}(?:corrupt|invalid)|PES packet size mismatch)", re.I)),
    ("packet_drop", re.compile(r"(?:dropp(?:ed|ing) packet|discard(?:ed|ing) packet|packet.{0,60}(?:dropped|discarded))", re.I)),
    ("decode_error", re.compile(r"(?:error while decoding|decoder error|decode error|decoding failed|error submitting packet to decoder|concealing.{0,80}errors? in|damaged macroblock|reference picture missing)", re.I)),
    ("timestamp_error", re.compile(r"(?:non[- ]monoton(?:ic|ous).{0,30}(?:DTS|PTS)|(?:DTS|PTS).{0,30}(?:out of order|invalid)|timestamp discontinuity)", re.I)),
    ("input_error", re.compile(r"(?:invalid data found when processing input|invalid argument|error opening input|error reading input|failed to read input|could not find codec parameters)", re.I)),
    ("mux_error", re.compile(r"(?:error muxing|error writing|failed to write|av_interleaved_write_frame|could not write header|error initializing output stream)", re.I)),
    ("io_error", re.compile(r"(?:connection reset|broken pipe|timed out|i/o error|input/output error|connection refused|unexpected end of file|premature EOF)", re.I)),
    ("other_error", re.compile(r"(?:\bfatal error\b|\berror while\b|\bfailed to\b|\bfailure\b)", re.I)),
)


def empty_error_fields():
    fields = {
        "ffmpeg_errors_total": "0",
        "ffmpeg_last_error_timestamp": "0",
        "ffmpeg_last_error_type": "none",
        "ffmpeg_stats_updated_timestamp": "0",
        "ffmpeg_frame": "0",
        "ffmpeg_fps": "0",
        "ffmpeg_speed": "0",
        "ffmpeg_output_bitrate_bits_per_second": "0",
    }
    fields.update({f"ffmpeg_error_{error_type}_total": "0" for error_type in ERROR_TYPES})
    fields.update({f"ffmpeg_restart_{reason}_total": "0" for reason in RESTART_REASONS})
    fields.update({f"ffmpeg_unexpected_stop_{reason}_total": "0" for reason in STOP_REASONS})
    fields.update({
        "ffmpeg_last_stop_timestamp": "0",
        "ffmpeg_last_exit_code": "unknown",
        "ffmpeg_writer_skipped_chunks_total": "0",
    })
    return fields


def initialize_ffmpeg_diagnostics(redis_client, key, *, reset=False, ttl=None):
    """Initialize bounded counters; optionally reset a dedicated diagnostics hash."""
    if not redis_client or not key:
        return
    try:
        if reset:
            redis_client.delete(key)
        fields = empty_error_fields()
        if reset:
            redis_client.hset(key, mapping=fields)
        else:
            for field, value in fields.items():
                redis_client.hsetnx(key, field, value)
            # A fresh process must not inherit a previous process's live stats.
            # Keep the counters so a fast restart between scrapes is observable.
            redis_client.hset(key, mapping={
                "ffmpeg_stats_updated_timestamp": "0",
                "ffmpeg_frame": "0",
                "ffmpeg_fps": "0",
                "ffmpeg_speed": "0",
                "ffmpeg_output_bitrate_bits_per_second": "0",
            })
        if ttl:
            redis_client.expire(key, int(ttl))
    except Exception:
        # Observability must never stop a stream or a transcode process.
        return


def classify_ffmpeg_error(line):
    text = str(line or "")
    for error_type, pattern in _ERROR_PATTERNS:
        if pattern.search(text):
            return error_type
    return None


def record_ffmpeg_error(redis_client, key, line):
    """Increment a fixed-category counter without storing the raw log line."""
    error_type = classify_ffmpeg_error(line)
    if not error_type or not redis_client or not key:
        return error_type
    try:
        pipe = redis_client.pipeline(transaction=False)
        pipe.hincrby(key, "ffmpeg_errors_total", 1)
        pipe.hincrby(key, f"ffmpeg_error_{error_type}_total", 1)
        pipe.hset(
            key,
            mapping={
                "ffmpeg_last_error_timestamp": str(time.time()),
                "ffmpeg_last_error_type": error_type,
            },
        )
        pipe.execute()
    except Exception:
        pass
    return error_type


def parse_ffmpeg_stats_line(line):
    """Parse common FFmpeg stderr progress fields into bounded numeric data."""
    text = str(line or "")
    result = {}
    patterns = (
        ("frame", r"\bframe=\s*(\d+)"),
        ("fps", r"\bfps=\s*([0-9.]+)"),
        ("speed", r"\bspeed=\s*([0-9.]+)x?"),
    )
    for field, pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            try:
                value = float(match.group(1))
                if math.isfinite(value):
                    result[field] = int(value) if field == "frame" else value
            except (ValueError, OverflowError):
                pass

    bitrate = re.search(r"\bbitrate=\s*([0-9.]+)\s*([kmg]?)bits/s", text, re.I)
    if bitrate:
        try:
            value = float(bitrate.group(1))
            scale = {"": 1, "k": 1000, "m": 1_000_000, "g": 1_000_000_000}
            value *= scale[bitrate.group(2).lower()]
            if math.isfinite(value):
                result["output_bitrate_bits_per_second"] = value
        except (ValueError, OverflowError):
            pass
    return result


def record_ffmpeg_stats(redis_client, key, stats, *, ttl=None):
    if not stats or not redis_client or not key:
        return
    fields = {f"ffmpeg_{name}": str(value) for name, value in stats.items()}
    fields["ffmpeg_stats_updated_timestamp"] = str(time.time())
    try:
        pipe = redis_client.pipeline(transaction=False)
        pipe.hset(key, mapping=fields)
        if ttl:
            pipe.expire(key, int(ttl))
        pipe.execute()
    except Exception:
        return


def record_output_process_event(redis_client, key, event, *, reason="other", exit_code=None, skipped=0):
    """Record an infrequent output-process transition with bounded fields."""
    if not redis_client or not key:
        return
    try:
        pipe = redis_client.pipeline(transaction=False)
        if event == "restart":
            if reason not in RESTART_REASONS:
                reason = "other"
            pipe.hincrby(key, f"ffmpeg_restart_{reason}_total", 1)
        elif event == "unexpected_stop":
            if reason not in STOP_REASONS:
                reason = "reader_error"
            pipe.hincrby(key, f"ffmpeg_unexpected_stop_{reason}_total", 1)
            code = str(exit_code) if isinstance(exit_code, int) else "unknown"
            pipe.hset(key, mapping={
                "ffmpeg_last_stop_timestamp": str(time.time()),
                "ffmpeg_last_exit_code": code,
            })
        elif event == "writer_skip" and skipped > 0:
            pipe.hincrby(key, "ffmpeg_writer_skipped_chunks_total", int(skipped))
        else:
            return
        pipe.execute()
    except Exception:
        # Metrics cannot interrupt an active transcode process.
        return
