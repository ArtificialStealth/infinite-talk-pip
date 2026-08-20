import json
import os
import shutil
import subprocess
import tempfile
import unittest

from composition_renderer import build_ass_captions, build_ffmpeg_command, validate_composition
from test_composition_renderer import composition


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg is required")
class FfmpegRenderIntegrationTests(unittest.TestCase):
    def test_renders_playable_portrait_mp4_with_overlay_and_captions(self):
        with tempfile.TemporaryDirectory() as directory:
            avatar = os.path.join(directory, "avatar.mp4")
            voice = os.path.join(directory, "voice.mp3")
            product = os.path.join(directory, "product.png")
            captions_path = os.path.join(directory, "captions.ass")
            output = os.path.join(directory, "final.mp4")
            subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=#445566:s=540x960:r=30:d=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", avatar], check=True, capture_output=True)
            subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", "libmp3lame", voice], check=True, capture_output=True)
            subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=#cc3344:s=200x200", "-frames:v", "1", product], check=True, capture_output=True)

            document = composition()
            document["canvas"]["durationMs"] = 1000
            for clip in document["clips"]:
                clip["startMs"] = 0
                clip["durationMs"] = 1000
            captions = {
                "words": [{"text": "Proof", "startMs": 0, "endMs": 800}],
                "phrases": [{"text": "Proof", "startMs": 0, "endMs": 800, "wordStart": 0, "wordEnd": 1}],
            }
            normalized = validate_composition(document, {"asset-1": "https://example.test/product.png"})
            with open(captions_path, "w", encoding="utf-8") as handle:
                handle.write(build_ass_captions(normalized, captions))
            command = build_ffmpeg_command(
                normalized,
                {"asset-1": {"path": product, "kind": "image"}},
                avatar, voice, None, output,
            )
            result = subprocess.run(command, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
            self.assertGreater(os.path.getsize(output), 1000)
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height", "-of", "json", output],
                check=True, capture_output=True, text=True,
            )
            metadata = json.loads(probe.stdout)
            streams = metadata["streams"]
            video = next(stream for stream in streams if stream["codec_type"] == "video")
            self.assertEqual((video["width"], video["height"]), (1080, 1920))
            self.assertTrue(any(stream["codec_type"] == "audio" for stream in streams))
            self.assertAlmostEqual(float(metadata["format"]["duration"]), 1.0, delta=1 / 30)


if __name__ == "__main__":
    unittest.main()
