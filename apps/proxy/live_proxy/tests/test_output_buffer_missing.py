"""Expired fMP4 fragments must be visible to delivery diagnostics."""

import unittest
from unittest.mock import patch

from apps.proxy.live_proxy.output.buffer import OutputStreamBuffer
from apps.proxy.live_proxy.redis_keys import RedisKeys


class _Pipeline:
    def __init__(self, values):
        self.values = values
        self.keys = []

    def get(self, key):
        self.keys.append(key)
        return self

    def execute(self):
        return [self.values.get(key) for key in self.keys]


class _Redis:
    def __init__(self, values):
        self.values = values

    def get(self, key):
        return self.values.get(key)

    def register_script(self, _script):
        return lambda **_kwargs: -1

    def pipeline(self, transaction=False):
        return _Pipeline(self.values)


class OutputBufferMissingTests(unittest.TestCase):
    def test_expired_middle_fragments_are_reported_without_changing_legacy_api(self):
        channel = "channel"
        fmt = "fmp4:p2"
        redis = _Redis({
            RedisKeys.output_buffer_index(channel, fmt): b"4",
            RedisKeys.output_buffer_chunk(channel, fmt, 1): b"one",
            RedisKeys.output_buffer_chunk(channel, fmt, 3): b"three",
        })
        with patch("apps.proxy.live_proxy.output.buffer.ConfigHelper.redis_chunk_ttl", return_value=60):
            buffer = OutputStreamBuffer(channel, redis_client=redis, fmt=fmt)

        chunks, head, missing = buffer.get_chunks_with_missing(start_index=0)
        self.assertEqual((chunks, head, missing), ([b"one", b"three"], 4, 2))
        self.assertEqual(buffer.get_chunks(start_index=0), ([b"one", b"three"], 4))


if __name__ == "__main__":
    unittest.main()
