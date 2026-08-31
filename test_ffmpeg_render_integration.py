import json
import os
import shutil
import subprocess
import tempfile
import unittest

from composition_renderer import build_ass_captions, build_ffmpeg_command, validate_composition
from test_composition_renderer import composition


FFMPEG = os.environ.get("FFMPEG_BINARY", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE_BINARY", "ffprobe")


@unittest.skipUnless(shutil.which(FFMPEG) and shutil.which(FFPROBE), "ffmpeg and ffprobe are required")
class FfmpegRenderIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = subprocess.run([FFMPEG, "-hide_banner", "-filters"], capture_output=True, text=True, timeout=15)
        if result.returncode != 0 or not any(" subtitles " in line for line in result.stdout.splitlines()):
            raise unittest.SkipTest("libass subtitles filter required; set FFMPEG_BINARY to a libass-enabled ffmpeg")

    def decode_frames(self, path, frames):
        result = subprocess.run([
            FFMPEG, "-v", "error", "-i", path, "-vf",
            "select='%s',scale=270:480" % "+".join("eq(n,%d)" % frame for frame in frames),
            "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
        ], check=True, capture_output=True, timeout=30)
        frame_size = 270 * 480
        self.assertEqual(len(result.stdout), len(frames) * frame_size)
        return [result.stdout[index * frame_size:(index + 1) * frame_size] for index in range(len(frames))]

    def test_renders_playable_portrait_mp4_with_overlay_and_captions(self):
        with tempfile.TemporaryDirectory() as directory:
            avatar = os.path.join(directory, "avatar.mp4")
            voice = os.path.join(directory, "voice.mp3")
            product = os.path.join(directory, "product.png")
            captions_path = os.path.join(directory, "captions.ass")
            output = os.path.join(directory, "final.mp4")
            subprocess.run([FFMPEG, "-y", "-f", "lavfi", "-i", "color=c=#445566:s=540x960:r=30:d=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", avatar], check=True, capture_output=True, timeout=30)
            subprocess.run([FFMPEG, "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", "libmp3lame", voice], check=True, capture_output=True, timeout=30)
            subprocess.run([FFMPEG, "-y", "-f", "lavfi", "-i", "color=c=#cc3344:s=200x200", "-frames:v", "1", product], check=True, capture_output=True, timeout=30)

            document = composition()
            document["canvas"]["durationMs"] = 1000
            document["captionStyle"]["fontSize"] = 100
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
                avatar, voice, captions_path, output,
            )
            command[0] = FFMPEG
            result = subprocess.run(command, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
            self.assertGreater(os.path.getsize(output), 1000)
            probe = subprocess.run(
                [FFPROBE, "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height", "-of", "json", output],
                check=True, capture_output=True, text=True, timeout=30,
            )
            metadata = json.loads(probe.stdout)
            streams = metadata["streams"]
            video = next(stream for stream in streams if stream["codec_type"] == "video")
            self.assertEqual((video["width"], video["height"]), (1080, 1920))
            self.assertTrue(any(stream["codec_type"] == "audio" for stream in streams))
            self.assertAlmostEqual(float(metadata["format"]["duration"]), 1.0, delta=1 / 30)
            frame = self.decode_frames(output, [5])[0]
            self.assertGreater(sum(pixel > 160 for pixel in frame[360 * 270:420 * 270]), 30,
                               "Expected burned caption pixels, not just a playable caption-free MP4")

    def test_scheduled_captions_move_on_exact_exported_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            avatar = os.path.join(directory, "avatar.mp4")
            voice = os.path.join(directory, "voice.wav")
            captions_path = os.path.join(directory, "captions.ass")
            output = os.path.join(directory, "scheduled.mp4")
            subprocess.run([FFMPEG, "-y", "-f", "lavfi", "-i", "color=c=black:s=540x960:r=30:d=1",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", avatar], check=True, capture_output=True, timeout=30)
            subprocess.run([FFMPEG, "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", voice],
                           check=True, capture_output=True, timeout=30)
            document = composition()
            document["canvas"]["durationMs"] = 1000
            document["clips"] = []
            document["captionStyle"].update(fontSize=100, activeWordColor="#ffffff")
            normalized = validate_composition(document, {})
            captions = {
                "words": [{"text": "Frame proof", "startMs": 0, "endMs": 1000}],
                "phrases": [{"startMs": 0, "endMs": 1000, "wordStart": 0, "wordEnd": 1}],
                "positionSchedule": {"version": 1, "fps": 30, "segments": [
                    {"startFrame": 0, "endFrame": 8, "x": 0.5, "y": 0.82},
                    {"startFrame": 8, "endFrame": 16, "x": 0.5, "y": 0.18},
                    {"startFrame": 16, "endFrame": 23, "x": 0.5, "y": 0.82},
                    {"startFrame": 23, "endFrame": 30, "x": 0.5, "y": 0.18},
                ]},
            }
            for highlight in [True, False]:
                with self.subTest(highlight=highlight):
                    normalized["captionStyle"]["highlightActiveWord"] = highlight
                    with open(captions_path, "w", encoding="utf-8") as handle:
                        handle.write(build_ass_captions(normalized, captions))
                    command = build_ffmpeg_command(normalized, {}, avatar, voice, captions_path, output)
                    command[0] = FFMPEG
                    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
                    self.assertEqual(result.returncode, 0, result.stderr[-2000:])
                    indices = [7, 8, 9, 15, 16, 17, 22, 23, 24]
                    for index, frame in zip(indices, self.decode_frames(output, indices)):
                        expected_y = 0.18 if 8 <= index < 16 or index >= 23 else 0.82
                        bright_rows = [offset // 270 for offset, pixel in enumerate(frame) if pixel > 200]
                        self.assertGreater(len(bright_rows), 50, (index, "no burned captions"))
                        midpoint = (min(bright_rows) + max(bright_rows)) / 2
                        self.assertAlmostEqual(midpoint, expected_y * 480, delta=6, msg="frame %d" % index)
                        self.assertLess(max(bright_rows) - min(bright_rows), 50,
                                        "both old and new caption positions visible on frame %d" % index)


if __name__ == "__main__":
    unittest.main()
