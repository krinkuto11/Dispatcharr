"""Check that the exporter discovers only bounded delivery diagnostic keys."""

import importlib.util
from pathlib import Path
import sys
import unittest


VENDOR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(VENDOR))
SPEC = importlib.util.spec_from_file_location("iptv_qoe_plugin", Path(__file__).with_name("plugin.py"))
PLUGIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLUGIN)


class _Redis:
    def scan_iter(self, match, count):
        return [
            b"qoe:channel:aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee:output:hls:p2:delivery:diagnostics",
            b"qoe:channel:aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee:output:token:delivery:diagnostics",
        ]

    def hgetall(self, key):
        return {b"hls_segment_expired_total": b"3"}


class DeliveryExporterTests(unittest.TestCase):
    def test_only_known_format_and_numeric_profile_are_exported(self):
        channel = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        diagnostics = PLUGIN.Plugin._output_delivery_diagnostics(_Redis(), channel)
        self.assertEqual(diagnostics, {"hls:p2": {"hls_segment_expired_total": "3"}})


if __name__ == "__main__":
    unittest.main()
