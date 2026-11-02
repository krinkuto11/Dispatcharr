import unittest

from probe import ProbeState


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


class ProbeStateTests(unittest.TestCase):
    def test_gap_is_counted_after_media_resumes(self):
        clock = Clock()
        state = ProbeState(stall_after=2.0, clock=clock)
        state.start_process()
        clock.value = 1.0
        state.media_progress(1_000_000)
        clock.value = 3.1
        state.tick()
        self.assertEqual(state.stall_started, 1.0)
        clock.value = 5.0
        state.media_progress(2_000_000)
        self.assertEqual(state.stalls_total, 1)
        self.assertEqual(state.stall_seconds_total, 4.0)
        self.assertIsNone(state.stall_started)

    def test_process_exit_closes_gap_and_timeline_reset_is_progress(self):
        clock = Clock()
        state = ProbeState(stall_after=2.0, clock=clock)
        state.start_process()
        clock.value = 1.0
        state.media_progress(4_000_000)
        clock.value = 2.0
        state.media_progress(100_000)
        self.assertEqual(state.timeline_resets_total, 1)
        clock.value = 5.0
        state.stop_process(1)
        self.assertEqual(state.stalls_total, 1)
        self.assertEqual(state.stall_seconds_total, 3.0)
        self.assertEqual(state.last_exit_code, 1)

    def test_slow_start_and_bounded_error_category(self):
        clock = Clock()
        state = ProbeState(startup_slow_after=10.0, clock=clock)
        state.start_process()
        clock.value = 12.0
        state.media_progress(2_000_000)
        state.record_error("error while decoding secret-url")
        self.assertEqual(state.startup_slow_total, 1)
        self.assertEqual(state.errors["decode"], 1)
        self.assertNotIn("secret-url", state.metrics())


if __name__ == "__main__":
    unittest.main()
