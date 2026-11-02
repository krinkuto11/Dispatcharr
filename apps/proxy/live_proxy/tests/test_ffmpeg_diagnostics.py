"""Check low-cardinality diagnostics emitted by live output processes."""

import unittest

from apps.proxy.live_proxy.ffmpeg_diagnostics import (
    empty_error_fields,
    initialize_ffmpeg_diagnostics,
    record_output_process_event,
)


class _Redis:
    def __init__(self):
        self.hashes = {}

    def pipeline(self, transaction=False):
        return self

    def hincrby(self, key, field, amount):
        values = self.hashes.setdefault(key, {})
        values[field] = int(values.get(field, 0)) + amount
        return self

    def hset(self, key, mapping):
        self.hashes.setdefault(key, {}).update(mapping)
        return self

    def hsetnx(self, key, field, value):
        self.hashes.setdefault(key, {}).setdefault(field, value)
        return self

    def expire(self, key, ttl):
        return True

    def execute(self):
        return []


class OutputProcessEventTests(unittest.TestCase):
    def test_fast_manager_restart_preserves_counters_and_clears_live_stats(self):
        redis = _Redis()
        key = "diagnostics"
        initialize_ffmpeg_diagnostics(redis, key, ttl=3600)
        record_output_process_event(redis, key, "restart", reason="pmt_layout_change")
        redis.hashes[key]["ffmpeg_stats_updated_timestamp"] = "123"
        redis.hashes[key]["ffmpeg_speed"] = "1.0"

        initialize_ffmpeg_diagnostics(redis, key, ttl=3600)

        self.assertEqual(redis.hashes[key]["ffmpeg_restart_pmt_layout_change_total"], 1)
        self.assertEqual(redis.hashes[key]["ffmpeg_stats_updated_timestamp"], "0")
        self.assertEqual(redis.hashes[key]["ffmpeg_speed"], "0")

    def test_restart_stop_and_skip_are_counted_with_fixed_fields(self):
        redis = _Redis()
        key = "diagnostics"
        redis.hashes[key] = empty_error_fields()

        record_output_process_event(redis, key, "restart", reason="input_discontinuity")
        record_output_process_event(redis, key, "restart", reason="untrusted source URL")
        record_output_process_event(redis, key, "unexpected_stop", reason="process_exit", exit_code=1)
        record_output_process_event(redis, key, "writer_skip", skipped=7)
        record_output_process_event(redis, key, "writer_skip", skipped=0)

        values = redis.hashes[key]
        self.assertEqual(values["ffmpeg_restart_input_discontinuity_total"], 1)
        self.assertEqual(values["ffmpeg_restart_other_total"], 1)
        self.assertEqual(values["ffmpeg_unexpected_stop_process_exit_total"], 1)
        self.assertEqual(values["ffmpeg_last_exit_code"], "1")
        self.assertEqual(values["ffmpeg_writer_skipped_chunks_total"], 7)
        self.assertNotIn("ffmpeg_restart_untrusted source URL_total", values)


if __name__ == "__main__":
    unittest.main()
