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

    def test_contain_fit_uses_padding_instead_of_crop(self):
        document = composition()
        document["clips"][1]["fitMode"] = "contain"
        normalized = validate_composition(document, {"asset-1": "https://example.test/product.png"})
        command = build_ffmpeg_command(
            normalized,
            {"asset-1": {"path": "/tmp/product.png", "kind": "image"}},
            "/tmp/avatar.mp4", "/tmp/voice.mp3", "/tmp/captions.ass", "/tmp/final.mp4",
        )
        filters = command[command.index("-filter_complex") + 1]
        self.assertIn("force_original_aspect_ratio=decrease,pad=324:324", filters)


class CaptionTests(unittest.TestCase):
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
