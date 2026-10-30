import unittest

import numpy as np

from streaming import StreamingTranscriber, finish_sentence, join_piece

SR = 16000


def seconds(n):
    return int(n * SR)


class FakeRecording:
    """Audio that grows as the test 'records', with known speech regions."""

    def __init__(self, regions):
        self.regions = regions  # absolute (start_s, end_s) speech
        self.length = 0

    def audio(self):
        return np.zeros(self.length, dtype=np.float32)

    def advance_to(self, s):
        self.length = seconds(s)


class StreamingTests(unittest.TestCase):
    def make(self, regions):
        rec = FakeRecording(regions)
        calls = []
        streamer = None

        def vad(pending):
            # Mimic Silero: regions relative to the pending audio; a region
            # only closes after a full 0.7 s pause, so until then (or while
            # still speaking) it runs to the end of the audio.
            offset = streamer._committed
            now = offset + pending.size
            out = []
            for start_s, end_s in rec.regions:
                start, end = seconds(start_s) - offset, seconds(end_s) - offset
                if end <= 0 or start >= pending.size:
                    continue
                closed = seconds(end_s) + seconds(0.7) <= now
                out.append({"start": max(0, start),
                            "end": end if closed else pending.size})
            return out

        def transcribe(audio, prompt):
            calls.append((round(audio.size / SR, 2), prompt))
            return f"piece{len(calls)}"

        streamer = StreamingTranscriber(rec.audio, transcribe, vad, sample_rate=SR)
        return rec, streamer, calls

    def test_cuts_at_a_finished_pause_and_leaves_ongoing_speech(self):
        rec, streamer, calls = self.make([(0.5, 3.0), (4.0, 9.0)])
        rec.advance_to(3.5)  # in the pause, but it isn't long enough yet
        streamer.step()
        self.assertEqual(calls, [])
        rec.advance_to(6.0)  # first region closed; second still going
        streamer.step()
        self.assertEqual(calls, [(3.0, None)])
        rec.advance_to(9.0)  # still talking at the very end: no cut
        streamer.step()
        self.assertEqual(len(calls), 1)

    def test_finish_transcribes_only_the_tail_with_prior_text_as_prompt(self):
        rec, streamer, calls = self.make([(0.5, 3.0), (4.0, 9.0)])
        rec.advance_to(6.0)
        streamer.step()
        rec.advance_to(9.0)
        text = streamer.finish(rec.audio())
        self.assertEqual(calls[-1], (6.0, "piece1"))
        self.assertEqual(text, "piece1 piece2.")

    def test_leading_silence_is_dropped_not_rescanned(self):
        rec, streamer, calls = self.make([])
        rec.advance_to(5.0)
        streamer.step()
        self.assertEqual(calls, [])
        self.assertGreater(streamer._committed, seconds(4.0))

    def test_without_pauses_everything_is_done_at_the_end(self):
        rec, streamer, calls = self.make([(0.2, 12.0)])
        for t in (2, 4, 8, 12):
            rec.advance_to(t)
            streamer.step()
        self.assertEqual(calls, [])
        self.assertEqual(streamer.finish(rec.audio()), "piece1")
        self.assertEqual(calls, [(12.0, None)])

    def test_failed_background_piece_is_redone_at_the_end(self):
        rec, streamer, calls = self.make([(0.5, 3.0), (4.0, 9.0)])

        def broken(audio, prompt):
            raise RuntimeError("gpu busy")

        streamer._transcribe = broken
        streamer._error = None
        rec.advance_to(6.0)
        with self.assertRaises(RuntimeError):
            streamer.step()
        self.assertEqual(streamer._committed, 0)  # nothing lost
        streamer._transcribe = lambda audio, prompt: "all of it"
        self.assertEqual(streamer.finish(rec.audio()), "all of it")

    def test_prompt_leaves_off_a_full_stop_the_cut_may_have_added(self):
        rec, streamer, calls = self.make([(0.5, 3.0), (4.0, 9.0)])
        streamer._transcribe = lambda audio, prompt: calls.append(prompt) or "And provide."
        rec.advance_to(6.0)
        streamer.step()
        streamer.finish(rec.audio())
        self.assertEqual(calls[-1], "And provide")


class JunctionTests(unittest.TestCase):
    def test_lowercase_continuation_drops_the_cut_full_stop(self):
        self.assertEqual(
            join_piece("Scan the New Regency.", "video files and provide."),
            "Scan the New Regency video files and provide.",
        )

    def test_capital_keeps_or_adds_the_full_stop(self):
        self.assertEqual(join_piece("It ends here.", "Then more."), "It ends here. Then more.")
        self.assertEqual(join_piece("until the end", "That is all."), "until the end. That is all.")

    def test_question_mark_is_kept_and_followed_by_a_capital(self):
        self.assertEqual(join_piece("Can we stream?", "so it starts"), "Can we stream? So it starts")

    def test_pronoun_i_does_not_start_a_sentence(self):
        self.assertEqual(join_piece("and then", "I left."), "and then I left.")

    def test_ellipses_at_the_cut_are_removed(self):
        self.assertEqual(join_piece("we could scan maybe...", "...the 50 most"), "we could scan maybe the 50 most")
        self.assertEqual(finish_sentence("all the way..."), "all the way.")

    def test_finish_sentence_adds_a_full_stop_only_when_missing(self):
        self.assertEqual(finish_sentence("run it"), "run it.")
        self.assertEqual(finish_sentence("run it?"), "run it?")


if __name__ == "__main__":
    unittest.main()
