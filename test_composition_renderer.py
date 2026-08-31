import copy
import math
import unittest

from composition_renderer import (
    CompositionValidationError,
    build_ass_captions,
    build_ffmpeg_command,
    validate_composition,
)


def composition():
    return {
        "schemaVersion": 1,
        "canvas": {
            "width": 1080,
            "height": 1920,
            "fps": 30,
            "durationMs": 15000,
            "background": "#000000",
        },
        "clips": [
            {
                "id": "avatar",
                "kind": "avatar",
                "assetId": None,
                "startMs": 0,
                "durationMs": 15000,
                "sourceOffsetMs": 0,
                "transform": {"x": 0, "y": 0, "width": 1080, "height": 1920, "rotation": 0},
                "fitMode": "cover",
                "mask": "rectangle",
                "opacity": 1,
                "zIndex": 20,
            },
            {
                "id": "product",
                "kind": "overlay",
                "assetId": "asset-1",
                "startMs": 5000,
                "durationMs": 4000,
                "sourceOffsetMs": 0,
                "transform": {"x": 65, "y": 1286, "width": 324, "height": 324, "rotation": 0},
                "fitMode": "cover",
                "mask": "rounded-rectangle",
                "opacity": 0.8,
                "zIndex": 40,
            },
            {
                "id": "captions",
                "kind": "caption",
                "assetId": None,
                "startMs": 0,
                "durationMs": 15000,
                "sourceOffsetMs": 0,
                "opacity": 1,
                "zIndex": 100,
            },
        ],
        "captionStyle": {
            "fontFamily": "Inter, sans-serif",
            "fontSize": 56,
            "fontWeight": 800,
            "color": "#ffffff",
            "activeWordColor": "#22d3ee",
            "backgroundColor": "#000000",
            "backgroundOpacity": 0.55,
            "outlineColor": "#000000",
            "outlineWidth": 2,
            "textAlign": "center",
            "uppercase": False,
            "highlightActiveWord": True,
            "maxWidth": 0.82,
            "position": {"x": 0.5, "y": 0.82},
        },
    }


class CompositionValidationTests(unittest.TestCase):
    def test_accepts_bounded_editor_document_and_referenced_assets(self):
        result = validate_composition(composition(), {"asset-1": "https://example.test/product.png"})
        self.assertEqual(result["canvas"]["width"], 1080)
        self.assertEqual([clip["id"] for clip in result["visualClips"]], ["avatar", "product"])

    def test_rejects_visual_clip_outside_canvas_duration(self):
        document = composition()
        document["clips"][1]["startMs"] = 14000
        document["clips"][1]["durationMs"] = 2000
        with self.assertRaisesRegex(CompositionValidationError, "clip timing"):
            validate_composition(document, {"asset-1": "https://example.test/product.png"})

    def test_rejects_missing_asset_reference(self):
        with self.assertRaisesRegex(CompositionValidationError, "asset reference"):
            validate_composition(composition(), {})

    def test_rejects_non_https_asset_url(self):
        with self.assertRaisesRegex(CompositionValidationError, "asset URL"):
            validate_composition(composition(), {"asset-1": "http://example.test/product.png"})


class FfmpegCompositionTests(unittest.TestCase):
    def test_uses_the_frame_aligned_voice_driven_canvas_duration(self):
        document = composition()
        document["canvas"]["durationMs"] = 16600
        for clip in document["clips"]:
            if clip["kind"] in {"avatar", "caption"}:
                clip["durationMs"] = 16600
        normalized = validate_composition(document, {"asset-1": "https://example.test/product.png"})

        command = build_ffmpeg_command(
            normalized,
            {"asset-1": {"path": "/tmp/product.png", "kind": "image"}},
            "/tmp/avatar.mp4", "/tmp/voice.mp3", "/tmp/captions.ass", "/tmp/final.mp4",
        )

        self.assertEqual(command[command.index("-t") + 1], "16.600")

    def test_builds_exact_timed_z_ordered_visual_filters_without_a_shell(self):
        document = validate_composition(composition(), {"asset-1": "https://example.test/product.png"})
        command = build_ffmpeg_command(
            document,
            {"asset-1": {"path": "/tmp/product.png", "kind": "image"}},
            avatar_path="/tmp/avatar.mp4",
            voice_path="/tmp/voice.mp3",
            captions_path="/tmp/captions.ass",
            output_path="/tmp/final.mp4",
        )
        self.assertEqual(command[0], "ffmpeg")
        self.assertNotIn("sh", command)
        filters = command[command.index("-filter_complex") + 1]
        self.assertLess(filters.index("[1:v]"), filters.index("[3:v]"))
        self.assertIn("scale=1080:1920:force_original_aspect_ratio=increase", filters)
        self.assertIn("scale=324:324:force_original_aspect_ratio=increase", filters)
        self.assertIn("enable='between(t,5.000,9.000)'", filters)
        self.assertIn("colorchannelmixer=aa=0.800", filters)
        self.assertIn("subtitles=filename='/tmp/captions.ass'", filters)
        self.assertIn("-crf", command)
        self.assertEqual(command[command.index("-crf") + 1], "18")

    def test_smaller_contain_fit_uses_transparent_padding_instead_of_crop(self):
        document = composition()
        document["clips"][1]["fitMode"] = "contain"
        normalized = validate_composition(document, {"asset-1": "https://example.test/product.png"})
        command = build_ffmpeg_command(
            normalized,
            {"asset-1": {"path": "/tmp/product.png", "kind": "image"}},
            "/tmp/avatar.mp4", "/tmp/voice.mp3", "/tmp/captions.ass", "/tmp/final.mp4",
        )
        filters = command[command.index("-filter_complex") + 1]
        self.assertIn("force_original_aspect_ratio=decrease,pad=324:324:(ow-iw)/2:(oh-ih)/2:color=black@0", filters)

    def test_fullscreen_contain_fit_uses_opaque_black_padding(self):
        document = composition()
        document["clips"][1]["fitMode"] = "contain"
        document["clips"][1]["transform"] = {
            "x": 0, "y": 0, "width": 1080, "height": 1920, "rotation": 0,
        }
        normalized = validate_composition(document, {"asset-1": "https://example.test/product.png"})
        command = build_ffmpeg_command(
            normalized,
            {"asset-1": {"path": "/tmp/product.png", "kind": "image"}},
            "/tmp/avatar.mp4", "/tmp/voice.mp3", "/tmp/captions.ass", "/tmp/final.mp4",
        )
        filters = command[command.index("-filter_complex") + 1]
        self.assertIn("force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black@1", filters)


class CaptionTests(unittest.TestCase):
    def scheduled_captions(self):
        return {
            "words": [
                {"text": "Exact", "startMs": 0, "endMs": 300},
                {"text": "voice", "startMs": 400, "endMs": 700},
            ],
            "phrases": [{"startMs": 0, "endMs": 700, "wordStart": 0, "wordEnd": 2}],
            "positionSchedule": {
                "version": 1, "fps": 30,
                "segments": [
                    {"startFrame": 0, "endFrame": 2, "x": 0.5, "y": 0.82},
                    {"startFrame": 2, "endFrame": 450, "x": 0.3, "y": 0.18},
                ],
            },
        }

    def events(self, document, captions):
        return [line for line in build_ass_captions(document, captions).splitlines() if line.startswith("Dialogue:")]

    def test_schedule_splits_inside_highlighted_word_and_keeps_phrase_and_styling(self):
        events = self.events(composition(), self.scheduled_captions())
        self.assertEqual(len(events), 4)
        self.assertIn("0:00:00.00,0:00:00.06", events[0])
        self.assertIn("0:00:00.06,0:00:00.30", events[1])
        self.assertIn(r"{\an5\pos(540,1574)}", events[0])
        self.assertIn(r"{\an5\pos(324,346)}", events[1])
        for event in events[:2]:
            self.assertIn(r"{\c&HEED322&}Exact {\c&HFFFFFF&}voice", event)

    def test_schedule_splits_unhighlighted_phrase(self):
        document = composition()
        document["captionStyle"]["highlightActiveWord"] = False
        document["captionStyle"]["uppercase"] = True
        events = self.events(document, self.scheduled_captions())
        self.assertEqual(len(events), 2)
        self.assertIn("0:00:00.00,0:00:00.06", events[0])
        self.assertIn("0:00:00.06,0:00:00.70", events[1])
        self.assertTrue(all(event.endswith("EXACT VOICE") for event in events))

    def test_schedule_keeps_unhighlighted_phrase_during_word_gaps(self):
        captions = self.scheduled_captions()
        captions["phrases"][0].update(startMs=0, endMs=900)
        captions["words"][0]["startMs"] = 100
        captions["positionSchedule"]["segments"][0]["endFrame"] = 10
        captions["positionSchedule"]["segments"][1]["startFrame"] = 10
        events = self.events(composition(), captions)
        for start, end in [("00", "10"), ("30", "33"), ("33", "40"), ("70", "90")]:
            matching = [event for event in events if "0:00:00.%s,0:00:00.%s" % (start, end) in event]
            self.assertEqual(len(matching), 1, (start, end, events))
            event = matching[0]
            self.assertIn(r"{\c&HFFFFFF&}Exact {\c&HFFFFFF&}voice", event)
            self.assertNotIn(r"\c&HEED322&", event)

    def test_schedule_boundaries_activate_on_exact_30fps_frames(self):
        captions = self.scheduled_captions()
        captions["positionSchedule"]["segments"] = [
            {"startFrame": frame, "endFrame": frame + 1, "x": 0.5, "y": 0.18 if frame % 2 else 0.82}
            for frame in range(20)
        ] + [{"startFrame": 20, "endFrame": 450, "x": 0.5, "y": 0.82}]
        events = self.events(composition(), captions)
        for frame in range(21):
            time_ms = math.floor(frame * 1000 / 30)
            def milliseconds(timestamp):
                hours, minutes, seconds = timestamp.split(":")
                return round((int(hours) * 3600 + int(minutes) * 60 + float(seconds)) * 1000)
            visible = [event for event in events if milliseconds(event.split(",")[1]) <= time_ms < milliseconds(event.split(",")[2])]
            self.assertEqual(len(visible), 1, (frame, visible))
            self.assertIn(r"\pos(540,%d)" % (346 if frame % 2 else 1574), visible[0])

    def test_schedule_uses_middle_vertical_anchor_for_each_text_alignment(self):
        for alignment, expected in [("left", 4), ("center", 5), ("right", 6)]:
            with self.subTest(alignment=alignment):
                document = composition()
                document["captionStyle"]["textAlign"] = alignment
                x = {"left": 97, "center": 540, "right": 983}[alignment]
                self.assertIn(r"\an%d\pos(%d,1574)" % (expected, x), self.events(document, self.scheduled_captions())[0])

    def test_absent_schedule_preserves_legacy_anchor_and_timing(self):
        captions = self.scheduled_captions()
        del captions["positionSchedule"]
        events = self.events(composition(), captions)
        self.assertEqual(len(events), 2)
        self.assertIn(r"\an2\pos(540,1574)", events[0])
        self.assertIn("0:00:00.00,0:00:00.30", events[0])
        self.assertIn("0:00:00.40,0:00:00.70", events[1])

    def test_schedule_covers_ceiling_of_fractional_frame_duration(self):
        document = composition()
        document["canvas"]["durationMs"] = 15001
        captions = self.scheduled_captions()
        with self.assertRaisesRegex(CompositionValidationError, "caption position schedule"):
            build_ass_captions(document, captions)
        captions["positionSchedule"]["segments"][-1]["endFrame"] = 451
        self.assertIn(r"\an5", build_ass_captions(document, captions))

    def test_rejects_malformed_supplied_schedule_even_without_phrases(self):
        valid = self.scheduled_captions()["positionSchedule"]
        invalid = [None, [], {}, {**valid, "version": 2}, {**valid, "version": True},
                   {**valid, "fps": 29.97}, {**valid, "fps": "30"},
                   {**valid, "segments": []}, {**valid, "segments": {}},
                   {**valid, "segments": [None]}]
        for key, value in [("startFrame", 1), ("startFrame", -1), ("startFrame", 0.5),
                           ("startFrame", False), ("endFrame", 0), ("endFrame", 3),
                           ("endFrame", 1), ("endFrame", 2.5), ("endFrame", True),
                           ("x", -0.1), ("x", 1.1), ("x", "0.5"), ("x", True),
                           ("y", float("nan")), ("y", float("inf")), ("y", None)]:
            schedule = copy.deepcopy(valid)
            schedule["segments"][0][key] = value
            invalid.append(schedule)
        invalid.append({**valid, "segments": valid["segments"][:-1]})
        invalid.append({**valid, "segments": list(reversed(valid["segments"]))})
        invalid.append({**valid, "segments": [valid["segments"][0], {**valid["segments"][1], "endFrame": 451}]})
        for schedule in invalid:
            with self.subTest(schedule=schedule):
                with self.assertRaisesRegex(CompositionValidationError, "caption position schedule"):
                    build_ass_captions(composition(), {"positionSchedule": schedule})

    def test_rejects_caption_words_beyond_the_effective_canvas(self):
        captions = {
            "words": [{"text": "cut", "startMs": 14900, "endMs": 16580}],
            "phrases": [{"text": "cut", "startMs": 14900, "endMs": 16580, "wordStart": 0, "wordEnd": 1}],
        }

        with self.assertRaisesRegex(CompositionValidationError, "caption timing"):
            build_ass_captions(
                validate_composition(composition(), {"asset-1": "https://example.test/product.png"}),
                captions,
            )

    def test_ass_uses_voice_word_timing_position_and_active_word_color(self):
        captions = {
            "words": [
                {"text": "Exact", "startMs": 0, "endMs": 300},
                {"text": "voice", "startMs": 320, "endMs": 700},
            ],
            "phrases": [
                {"text": "Exact voice", "startMs": 0, "endMs": 700, "wordStart": 0, "wordEnd": 2},
            ],
        }
        ass = build_ass_captions(validate_composition(composition(), {"asset-1": "https://example.test/product.png"}), captions)
        self.assertIn("PlayResX: 1080", ass)
        self.assertIn("Dialogue: 0,0:00:00.00,0:00:00.30", ass)
        self.assertIn(r"\pos(540,1574)", ass)
        self.assertIn(r"{\c&HEED322&}Exact", ass)
        self.assertIn(r"Exact {\c&HFFFFFF&}voice", ass)


if __name__ == "__main__":
    unittest.main()
