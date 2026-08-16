import pathlib
import unittest


class HandlerContractTests(unittest.TestCase):
    def test_dispatches_dynamic_editor_compositions_after_avatar_generation(self):
        source = pathlib.Path("handler.py").read_text()
        self.assertIn('render_type == "ugc_composition_v1"', source)
        self.assertIn("validate_composition(job_input.get(\"composition\")", source)
        self.assertIn("build_ass_captions", source)
        self.assertIn("build_ffmpeg_command", source)

    def test_production_dockerfile_copies_composition_module(self):
        source = pathlib.Path("Dockerfile").read_text()
        self.assertIn("COPY . .", source)
        self.assertTrue(pathlib.Path("composition_renderer.py").exists())


if __name__ == "__main__":
    unittest.main()
