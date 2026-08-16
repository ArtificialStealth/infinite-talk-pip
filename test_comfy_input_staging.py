import pathlib
import tempfile
import unittest

from comfy_input_staging import cleanup_comfy_inputs, stage_comfy_inputs


class ComfyInputStagingTests(unittest.TestCase):
    def test_copies_job_media_under_comfy_input_and_returns_relative_names(self):
        with tempfile.TemporaryDirectory() as root:
            source = pathlib.Path(root) / "source"
            source.mkdir()
            media = source / "avatar.png"
            audio = source / "voice.wav"
            media.write_bytes(b"image")
            audio.write_bytes(b"audio")
            input_root = pathlib.Path(root) / "comfy-input"

            staged = stage_comfy_inputs(
                str(media), [str(audio)], "task_1234-abcd", input_root=str(input_root)
            )

            self.assertEqual(staged.media_reference, "task_1234-abcd/media.png")
            self.assertEqual(staged.audio_references, ["task_1234-abcd/audio_1.wav"])
            self.assertEqual((input_root / staged.media_reference).read_bytes(), b"image")
            self.assertEqual((input_root / staged.audio_references[0]).read_bytes(), b"audio")

            cleanup_comfy_inputs(staged.directory, input_root=str(input_root))
            self.assertFalse(pathlib.Path(staged.directory).exists())

    def test_rejects_an_unsafe_task_identifier(self):
        with tempfile.TemporaryDirectory() as root:
            source = pathlib.Path(root) / "avatar.png"
            source.write_bytes(b"image")
            with self.assertRaises(ValueError):
                stage_comfy_inputs(str(source), [], "../escape", input_root=root)


if __name__ == "__main__":
    unittest.main()
