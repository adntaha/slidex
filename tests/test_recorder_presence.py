import json
import unittest
from unittest.mock import Mock, patch

from api import slides
import slidex


class RecorderPresenceTests(unittest.TestCase):
    def setUp(self):
        self.store = patch.object(slides, "_fallback", {})
        self.credentials = patch.object(slides, "credentials", return_value=None)
        self.clock = patch.object(slides.time, "time", return_value=1000)
        self.store.start()
        self.credentials.start()
        self.now = self.clock.start()
        self.addCleanup(self.store.stop)
        self.addCleanup(self.credentials.stop)
        self.addCleanup(self.clock.stop)

    def test_empty_or_legacy_deck_is_offline(self):
        self.assertFalse(slides.recorder_connected(slides.read_deck()))
        self.assertFalse(slides.recorder_connected({"slides": [{"id": 1}]}))

    def test_silent_or_paused_recorder_expires_and_reconnects(self):
        deck = {"version": 1, "paused": True, "slides": [{"id": 1}]}
        slides.write_deck(deck)
        self.now.return_value = 1014
        self.assertTrue(slides.recorder_connected(slides.read_deck()))
        self.now.return_value = 1015
        self.assertFalse(slides.recorder_connected(slides.read_deck()))
        self.assertEqual(slides.read_deck()["slides"], deck["slides"])
        slides.write_deck(deck)
        self.assertTrue(slides.recorder_connected(slides.read_deck()))

    def test_presence_uses_server_receipt_time(self):
        slides.write_deck({"slides": [], "recorder_seen_at": 99999999})
        self.assertEqual(slides.read_deck()["recorder_seen_at"], 1000)

    def test_old_publisher_refreshes_presence_without_overwriting_newer_slides(self):
        slides.write_deck({"version": 20, "slides": [{"id": 1}]})
        self.now.return_value = 1020
        self.assertFalse(slides.write_deck({"version": 0, "slides": []}))
        self.assertEqual(slides.read_deck()["version"], 20)
        self.assertEqual(slides.read_deck()["slides"], [{"id": 1}])
        self.assertTrue(slides.recorder_connected(slides.read_deck()))

    def test_get_overrides_stored_connected_flag_after_expiry(self):
        slides.write_deck({"slides": [], "recorder_connected": True})
        self.now.return_value = 1015
        request = object.__new__(slides.handler)
        request._send = Mock()
        request.do_GET()
        status, payload = request._send.call_args.args
        self.assertEqual(status, 200)
        self.assertFalse(payload["recorder_connected"])

    def test_publisher_sends_unchanged_deck_heartbeat(self):
        stop = Mock()
        stop.wait.side_effect = [False, False, False, True]
        thread = Mock()
        deck = {"slides": [], "paused": True}
        with patch.object(slidex.threading, "Event", return_value=stop), \
             patch.object(slidex.threading, "Thread", return_value=thread) as make_thread, \
             patch.object(slidex, "deck_snapshot", return_value=deck), \
             patch.object(slidex.time, "monotonic", side_effect=[100, 101, 105, 105]), \
             patch.object(slidex, "urlopen") as send, \
             patch.object(slidex, "trace"):
            slidex.start_deck_publisher("https://example.com/api/slides", "test")
            make_thread.call_args.kwargs["target"]()
        self.assertEqual(send.call_count, 2)
        for call in send.call_args_list:
            self.assertEqual(json.loads(call.args[0].data), deck)


if __name__ == "__main__":
    unittest.main()
