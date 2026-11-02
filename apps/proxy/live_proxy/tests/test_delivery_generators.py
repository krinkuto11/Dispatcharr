"""Output generators report actual server-side waits and missing media."""

from unittest.mock import MagicMock, patch

from django.test import SimpleTestCase

from apps.proxy.live_proxy.output.fmp4.generator import FMP4StreamGenerator
from apps.proxy.live_proxy.output.ts.generator import StreamGenerator


class DeliveryGeneratorTests(SimpleTestCase):
    def test_ts_wait_after_first_chunk_is_playback_gap(self):
        generator = StreamGenerator.__new__(StreamGenerator)
        generator.channel_id = "channel"
        generator.fmt = "mpegts:p2"
        generator.chunks_sent = 4
        generator._delivery_wait_started = None
        generator._delivery_wait_keepalives = 0
        generator.proxy_server = MagicMock()

        with patch("apps.proxy.live_proxy.output.ts.generator.time.monotonic", side_effect=[10.0, 13.0]), \
             patch("apps.proxy.live_proxy.output.ts.generator.record_starvation") as record:
            generator._start_delivery_wait()
            generator._finish_delivery_wait("recovered")

        record.assert_called_once_with(
            generator.proxy_server.redis_client, "channel", "mpegts:p2",
            3.0, "playback", "recovered", keepalives=0,
        )

    def test_fmp4_missing_fragments_are_counted_before_wait(self):
        generator = FMP4StreamGenerator.__new__(FMP4StreamGenerator)
        generator.channel_id = "channel"
        generator.fmt = "fmp4"
        generator.local_index = 1
        generator.chunks_sent = 1
        generator.consecutive_empty = 0
        generator._delivery_wait_started = None
        generator.fmp4_buffer = MagicMock()
        generator.fmp4_buffer.get_chunks_with_missing.return_value = ([], 4, 3)
        generator.proxy_server = MagicMock()

        with patch.object(generator, "_check_resources", return_value=True), \
             patch.object(generator, "_is_timeout", return_value=True), \
             patch("apps.proxy.live_proxy.output.fmp4.generator.gevent.sleep"), \
             patch("apps.proxy.live_proxy.output.fmp4.generator.record_skipped_chunks") as record:
            self.assertEqual(list(generator._stream_data_generator()), [])

        record.assert_called_once_with(generator.proxy_server.redis_client, "channel", "fmp4", 3)
        self.assertEqual(generator.local_index, 4)
