"""Tests for bounded, event-based live delivery metrics."""

import unittest

from apps.proxy.live_proxy.delivery_diagnostics import (
    record_hls_event,
    record_hls_segment,
    record_skipped_chunks,
    record_starvation,
)
from apps.proxy.live_proxy.redis_keys import RedisKeys


class _Redis:
    def __init__(self):
        self.hashes = {}

    def pipeline(self, transaction=False):
        return self

    def hget(self, key, field):
        return self.hashes.get(key, {}).get(field)

    def hincrby(self, key, field, amount):
        values = self.hashes.setdefault(key, {})
        values[field] = int(values.get(field, 0)) + amount
        return self

    def hincrbyfloat(self, key, field, amount):
        values = self.hashes.setdefault(key, {})
        values[field] = float(values.get(field, 0)) + amount
        return self

    def hset(self, key, mapping):
        self.hashes.setdefault(key, {}).update(mapping)
        return self

    def expire(self, key, ttl):
        return True

    def execute(self):
        return []


class DeliveryDiagnosticsTests(unittest.TestCase):
    def test_short_wait_is_ignored_and_long_wait_is_counted_by_phase(self):
        redis = _Redis()
        key = RedisKeys.output_delivery_diagnostics("channel", "mpegts:p2")
        record_starvation(redis, "channel", "mpegts:p2", 1.9, "playback", "recovered")
        self.assertNotIn(key, redis.hashes)

        record_starvation(redis, "channel", "mpegts:p2", 3.25, "playback", "recovered", keepalives=2)
        self.assertEqual(redis.hashes[key]["starvation_playback_recovered_total"], 1)
        self.assertEqual(redis.hashes[key]["starvation_playback_seconds_total"], 3.25)
        self.assertEqual(redis.hashes[key]["keepalive_packets_total"], 2)

    def test_format_and_event_vocabulary_cannot_expand_keyspace(self):
        redis = _Redis()
        record_skipped_chunks(redis, "channel", "mpegts:p1:token", 4)
        record_hls_event(redis, "channel", "hls", "private-url")
        self.assertFalse(redis.hashes)
        record_skipped_chunks(redis, "channel", "fmp4", 3)
        record_hls_event(redis, "channel", "hls", "segment_advertised_missing")
        self.assertEqual(len(redis.hashes), 2)

    def test_hls_publish_gap_uses_advertised_target(self):
        redis = _Redis()
        key = RedisKeys.output_delivery_diagnostics("channel", "hls:p1")
        record_hls_segment(redis, "channel", "hls:p1", 100.0, 6.0)
        record_hls_segment(redis, "channel", "hls:p1", 104.0, 6.0)
        record_hls_segment(redis, "channel", "hls:p1", 114.0, 6.0)
        self.assertEqual(redis.hashes[key]["hls_segments_published_total"], 3)
        self.assertEqual(redis.hashes[key]["hls_segment_late_total"], 1)
        self.assertEqual(redis.hashes[key]["hls_last_publish_gap_seconds"], "10.0")


if __name__ == "__main__":
    unittest.main()
