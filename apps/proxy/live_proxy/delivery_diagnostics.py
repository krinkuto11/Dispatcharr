"""Low-cardinality, best-effort diagnostics for live output delivery.

Redis writes happen when an anomalous episode closes, a chunk is skipped, an
HLS request fails, or a segment is published. No write occurs per TS packet.
"""

import math
from collections import deque
from statistics import median
import time

from .redis_keys import RedisKeys


DIAGNOSTICS_TTL = 86400
MIN_STARVATION_SECONDS = 2.0


class DeliveryGapTracker:
    """Ignore normal chunk cadence while retaining a two-second floor.

    TS/fMP4 buffers publish complete chunks, so low-bitrate streams may
    normally wait longer than two seconds between chunks. Learn a short
    per-client baseline and count only waits well above that cadence.
    """

    def __init__(self):
        self._normal_waits = deque(maxlen=6)

    def is_anomalous(self, seconds):
        try:
            seconds = float(seconds)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(seconds) or seconds < 0:
            return False
        threshold = (
            max(MIN_STARVATION_SECONDS, 1.8 * median(self._normal_waits))
            if len(self._normal_waits) >= 3 else 5.0
        )
        if seconds >= threshold:
            return True
        if seconds >= 0.05:
            self._normal_waits.append(seconds)
        return False
PHASES = ("startup", "playback")
OUTCOMES = ("recovered", "ended")
HLS_EVENTS = (
    "playlist_stale", "playlist_malformed", "playlist_cold_start",
    "playlist_cold_start_timeout", "segment_expired",
    "segment_advertised_missing",
)


def _valid_fmt(fmt):
    """Restrict Redis key cardinality to known output types and numeric profiles."""
    parts = str(fmt or "").split(":")
    if not parts or parts[0] not in ("mpegts", "fmp4", "hls"):
        return None
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2 and 1 < len(parts[1]) <= 11 and parts[1].startswith("p") and parts[1][1:].isdigit():
        return f"{parts[0]}:p{int(parts[1][1:])}"
    return None


def _key(channel_id, fmt):
    safe_fmt = _valid_fmt(fmt)
    if not channel_id or not safe_fmt:
        return None
    return RedisKeys.output_delivery_diagnostics(channel_id, safe_fmt)


def _execute(redis_client, key, increments, values=None):
    if not redis_client or not key:
        return
    try:
        pipe = redis_client.pipeline(transaction=False)
        for field, amount in increments.items():
            if isinstance(amount, float):
                pipe.hincrbyfloat(key, field, amount)
            else:
                pipe.hincrby(key, field, amount)
        if values:
            pipe.hset(key, mapping=values)
        pipe.expire(key, DIAGNOSTICS_TTL)
        pipe.execute()
    except Exception:
        # A metrics backend failure must never interrupt a video response.
        return


def record_starvation(redis_client, channel_id, fmt, seconds, phase, outcome, *, keepalives=0):
    """Count one caller-qualified server data gap after it closes or ends."""
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return
    if not math.isfinite(seconds) or seconds < MIN_STARVATION_SECONDS:
        return
    phase = phase if phase in PHASES else "playback"
    outcome = outcome if outcome in OUTCOMES else "ended"
    increments = {
        f"starvation_{phase}_{outcome}_total": 1,
        f"starvation_{phase}_seconds_total": seconds,
    }
    if keepalives > 0:
        increments["keepalive_packets_total"] = int(keepalives)
    _execute(redis_client, _key(channel_id, fmt), increments, {
        "last_starvation_timestamp": str(time.time()),
        "last_starvation_seconds": str(seconds),
    })


def record_skipped_chunks(redis_client, channel_id, fmt, count):
    """Count output chunks that the client had to jump over."""
    try:
        count = int(count)
    except (TypeError, ValueError):
        return
    if count > 0:
        _execute(redis_client, _key(channel_id, fmt), {"skipped_chunks_total": count})


def record_hls_event(redis_client, channel_id, fmt, event):
    if event in HLS_EVENTS:
        _execute(redis_client, _key(channel_id, fmt), {f"hls_{event}_total": 1})


def record_hls_segment(redis_client, channel_id, fmt, published_at, target_seconds):
    """Track segment publish cadence; avoid a read/modify/write per segment."""
    key = _key(channel_id, fmt)
    if not redis_client or not key:
        return
    try:
        published_at = float(published_at)
        target_seconds = float(target_seconds)
        if not math.isfinite(published_at) or not math.isfinite(target_seconds) or target_seconds <= 0:
            return
        previous = redis_client.hget(key, "hls_last_segment_timestamp")
        gap = max(0.0, published_at - float(previous)) if previous else 0.0
        increments = {"hls_segments_published_total": 1}
        if previous and gap > 1.5 * target_seconds:
            increments["hls_segment_late_total"] = 1
        _execute(redis_client, key, increments, {
            "hls_last_segment_timestamp": str(published_at),
            "hls_last_publish_gap_seconds": str(gap),
        })
    except Exception:
        return
