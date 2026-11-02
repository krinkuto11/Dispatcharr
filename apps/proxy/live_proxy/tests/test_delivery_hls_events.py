"""The HLS error counters distinguish expired from advertised missing media."""

import json
from unittest.mock import MagicMock, call, patch

from django.test import SimpleTestCase

from apps.proxy.live_proxy.output.hls import views


CHANNEL = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


class HLSDeliveryEventTests(SimpleTestCase):
    @patch("apps.proxy.live_proxy.output.hls.views.record_hls_event")
    @patch("apps.proxy.live_proxy.output.hls.views.RedisClient.get_buffer")
    @patch("apps.proxy.live_proxy.output.hls.views._load_live_session")
    @patch("apps.proxy.live_proxy.output.hls.views.ProxyServer.get_instance")
    def test_advertised_segment_missing_counts_both_cause_and_http_failure(
        self, get_instance, load_session, get_buffer, record_event,
    ):
        redis = MagicMock()
        redis.get.return_value = json.dumps({"window": [{"seq": 3}]}).encode()
        get_instance.return_value.redis_client = redis
        load_session.return_value = ((CHANNEL, "client", {"output_profile_id": ""}), None)
        get_buffer.return_value.get.return_value = None

        response = views._serve_hls_segment("token", 3)

        self.assertEqual(response.status_code, 404)
        record_event.assert_has_calls([
            call(redis, CHANNEL, "hls", "segment_expired"),
            call(redis, CHANNEL, "hls", "segment_advertised_missing"),
        ])

    @patch("apps.proxy.live_proxy.output.hls.views.record_hls_event")
    @patch("apps.proxy.live_proxy.output.hls.views._load_live_session")
    @patch("apps.proxy.live_proxy.output.hls.views.ProxyServer.get_instance")
    def test_stale_playlist_counts_error(self, get_instance, load_session, record_event):
        redis = MagicMock()
        redis.get.return_value = b'{"window":[],"ts":1,"adv_target":4}'
        get_instance.return_value.redis_client = redis
        load_session.return_value = ((CHANNEL, "client", {"output_profile_id": ""}), None)

        response = views._serve_hls_playlist("token")

        self.assertEqual(response.status_code, 404)
        record_event.assert_called_once_with(redis, CHANNEL, "hls", "playlist_stale")
