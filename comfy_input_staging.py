"""Per-job staging for ComfyUI file-loader nodes."""

from dataclasses import dataclass
import os
import re
import shutil


SAFE_TASK_ID = re.compile(r"^task_[0-9a-f-]+$")


@dataclass(frozen=True)
class StagedComfyInputs:
    directory: str
    media_reference: str
    audio_references: list


def _extension(path, fallback):
    suffix = os.path.splitext(path)[1].lower()
    return suffix if re.fullmatch(r"\.[a-z0-9]{1,8}", suffix or "") else fallback


def stage_comfy_inputs(media_path, audio_paths, task_id, input_root="/ComfyUI/input"):
    """Copy trusted downloaded inputs into the directory required by ComfyUI loaders."""
    if not SAFE_TASK_ID.fullmatch(task_id):
        raise ValueError("invalid ComfyUI staging task id")
    directory = os.path.join(input_root, task_id)
    os.makedirs(directory, exist_ok=False)
    media_name = "media" + _extension(media_path, ".bin")
    shutil.copy2(media_path, os.path.join(directory, media_name))
    audio_references = []
    for index, audio_path in enumerate(audio_paths, start=1):
        audio_name = "audio_%d%s" % (index, _extension(audio_path, ".wav"))
        shutil.copy2(audio_path, os.path.join(directory, audio_name))
        audio_references.append("%s/%s" % (task_id, audio_name))
    return StagedComfyInputs(
        directory=directory,
        media_reference="%s/%s" % (task_id, media_name),
        audio_references=audio_references,
    )


def cleanup_comfy_inputs(directory, input_root="/ComfyUI/input"):
    """Remove only a validated immediate child of the configured ComfyUI input root."""
    root = os.path.realpath(input_root)
    target = os.path.realpath(directory)
    if os.path.dirname(target) != root or not SAFE_TASK_ID.fullmatch(os.path.basename(target)):
        raise ValueError("unsafe ComfyUI staging cleanup")
    shutil.rmtree(target, ignore_errors=True)
