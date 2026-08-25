"""Validated FFmpeg rendering for Fireship UGC editor compositions."""

import copy
import math
import re
from urllib.parse import urlsplit


ALLOWED_CANVASES = {(1080, 1920), (1920, 1080), (1080, 1080)}
VISUAL_KINDS = {"background", "avatar", "overlay"}
FIT_MODES = {"contain", "cover", "fit-width", "fit-height", "stretch", "free"}
MASKS = {"none", "rectangle", "rounded-rectangle", "square", "circle"}
HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")


class CompositionValidationError(ValueError):
    pass


def _number(value, label, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise CompositionValidationError("invalid %s" % label)
    if minimum is not None and value < minimum:
        raise CompositionValidationError("invalid %s" % label)
    if maximum is not None and value > maximum:
        raise CompositionValidationError("invalid %s" % label)
    return value


def _color(value, fallback="#000000"):
    return value if isinstance(value, str) and HEX_COLOR.match(value) else fallback


def validate_composition(document, assets):
    if not isinstance(document, dict) or document.get("schemaVersion") != 1:
        raise CompositionValidationError("invalid composition schema")
    canvas = document.get("canvas")
    clips = document.get("clips")
    if not isinstance(canvas, dict) or not isinstance(clips, list) or len(clips) > 100:
        raise CompositionValidationError("invalid composition structure")
    width = int(_number(canvas.get("width"), "canvas width", 1, 4096))
    height = int(_number(canvas.get("height"), "canvas height", 1, 4096))
    if (width, height) not in ALLOWED_CANVASES or canvas.get("fps") != 30:
        raise CompositionValidationError("unsupported canvas")
    duration_ms = int(_number(canvas.get("durationMs"), "canvas duration", 1000, 60000))
    if not isinstance(assets, dict) or len(assets) > 50:
        raise CompositionValidationError("invalid assets")
    for url in assets.values():
        if not isinstance(url, str) or len(url) > 4096:
            raise CompositionValidationError("invalid asset URL")
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise CompositionValidationError("invalid asset URL")

    normalized = copy.deepcopy(document)
    normalized["canvas"]["background"] = _color(canvas.get("background"))
    visual_clips = []
    music_clips = []
    for index, clip in enumerate(clips):
        if not isinstance(clip, dict) or not isinstance(clip.get("id"), str):
            raise CompositionValidationError("invalid clip")
        start_ms = int(_number(clip.get("startMs"), "clip timing", 0, duration_ms))
        clip_duration = int(_number(clip.get("durationMs"), "clip timing", 1, duration_ms))
        if start_ms + clip_duration > duration_ms:
            raise CompositionValidationError("invalid clip timing")
        kind = clip.get("kind")
        if kind in VISUAL_KINDS:
            transform = clip.get("transform")
            if not isinstance(transform, dict):
                raise CompositionValidationError("invalid clip transform")
            for field in ("x", "y", "width", "height", "rotation"):
                _number(transform.get(field), "clip transform", -8192, 8192)
            if transform["width"] <= 0 or transform["height"] <= 0:
                raise CompositionValidationError("invalid clip transform")
            fit_mode = clip.get("fitMode", "cover")
            mask = clip.get("mask", "rectangle")
            if fit_mode not in FIT_MODES or mask not in MASKS:
                raise CompositionValidationError("invalid clip appearance")
            _number(clip.get("opacity", 1), "clip opacity", 0, 1)
            _number(clip.get("zIndex", 0), "clip z-index", -1000, 1000)
            if kind != "avatar":
                asset_id = clip.get("assetId")
                if not isinstance(asset_id, str) or asset_id not in assets:
                    raise CompositionValidationError("missing asset reference")
            visual_clips.append((clip.get("zIndex", 0), index, normalized["clips"][index]))
        elif kind == "music":
            asset_id = clip.get("assetId")
            if not isinstance(asset_id, str) or asset_id not in assets:
                raise CompositionValidationError("missing asset reference")
            _number(clip.get("volume", 1), "music volume", 0, 2)
            music_clips.append(normalized["clips"][index])
    normalized["visualClips"] = [item[2] for item in sorted(visual_clips)]
    normalized["musicClips"] = music_clips
    return normalized


def _ass_time(milliseconds):
    centiseconds = max(0, int(round(milliseconds / 10.0)))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    seconds, fraction = divmod(remainder, 100)
    return "%d:%02d:%02d.%02d" % (hours, minutes, seconds, fraction)


def _ass_color(value):
    color = _color(value, "#ffffff")[1:].upper()
    return color[4:6] + color[2:4] + color[0:2]


def _ass_alpha(opacity):
    bounded = min(1.0, max(0.0, float(opacity)))
    return "%02X" % round((1.0 - bounded) * 255)


def _ass_text(value):
    return str(value).replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}").replace("\n", "\\N")


def build_ass_captions(composition, captions):
    canvas = composition["canvas"]
    style = composition.get("captionStyle") or {}
    width, height = canvas["width"], canvas["height"]
    position = style.get("position") if isinstance(style.get("position"), dict) else {}
    x = round(_number(position.get("x", 0.5), "caption x", 0, 1) * width)
    y = round(_number(position.get("y", 0.82), "caption y", 0, 1) * height)
    text_align = style.get("textAlign", "center")
    alignment = {"left": 1, "center": 2, "right": 3}.get(text_align, 2)
    font_family = str(style.get("fontFamily", "Inter")).split(",", 1)[0].strip()
    font_family = re.sub(r"[^A-Za-z0-9 _-]", "", font_family)[:64] or "Inter"
    font_size = int(_number(style.get("fontSize", 56), "caption font size", 12, 240))
    weight = int(_number(style.get("fontWeight", 800), "caption font weight", 100, 900))
    primary = _ass_color(style.get("color", "#ffffff"))
    active = _ass_color(style.get("activeWordColor", "#22d3ee"))
    outline = _ass_color(style.get("outlineColor", "#000000"))
    background = _ass_color(style.get("backgroundColor", "#000000"))
    back_alpha = _ass_alpha(style.get("backgroundOpacity", 0.55))
    outline_width = float(_number(style.get("outlineWidth", 2), "caption outline", 0, 12))
    max_width = _number(style.get("maxWidth", 0.82), "caption max width", 0.2, 1)
    margin = round(width * (1 - max_width) / 2)
    uppercase = style.get("uppercase") is True
    highlight = style.get("highlightActiveWord") is not False
    header = "\n".join([
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: %d" % width,
        "PlayResY: %d" % height,
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Captions,%s,%d,&H00%s,&H00%s,&H00%s,&H%s%s,%d,0,0,0,100,100,0,0,3,%.1f,0,%d,%d,%d,0,1" % (
            font_family, font_size, primary, active, outline, back_alpha, background,
            -1 if weight >= 600 else 0, outline_width, alignment, margin, margin,
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ])
    words = captions.get("words", []) if isinstance(captions, dict) else []
    phrases = captions.get("phrases", []) if isinstance(captions, dict) else []
    duration_ms = canvas["durationMs"]
    for item in words + phrases:
        if not isinstance(item, dict):
            raise CompositionValidationError("invalid caption timing")
        start_ms = _number(item.get("startMs"), "caption timing", 0, duration_ms)
        end_ms = _number(item.get("endMs"), "caption timing", 0, duration_ms)
        if end_ms <= start_ms:
            raise CompositionValidationError("invalid caption timing")
    events = []
    for phrase in phrases:
        if not isinstance(phrase, dict):
            continue
        start_index = int(phrase.get("wordStart", 0))
        end_index = int(phrase.get("wordEnd", 0))
        phrase_words = words[start_index:end_index]
        if not phrase_words:
            continue
        rendered_words = [str(item.get("text", "")).upper() if uppercase else str(item.get("text", "")) for item in phrase_words]
        if highlight:
            for active_index, word in enumerate(phrase_words):
                pieces = []
                for index, text in enumerate(rendered_words):
                    color = active if index == active_index else primary
                    pieces.append(r"{\c&H%s&}%s" % (color, _ass_text(text)))
                event_text = r"{\an%d\pos(%d,%d)}" % (alignment, x, y) + " ".join(pieces)
                events.append("Dialogue: 0,%s,%s,Captions,,0,0,0,,%s" % (
                    _ass_time(word.get("startMs", 0)), _ass_time(word.get("endMs", 0)), event_text,
                ))
        else:
            event_text = r"{\an%d\pos(%d,%d)}" % (alignment, x, y) + _ass_text(" ".join(rendered_words))
            events.append("Dialogue: 0,%s,%s,Captions,,0,0,0,,%s" % (
                _ass_time(phrase.get("startMs", 0)), _ass_time(phrase.get("endMs", 0)), event_text,
            ))
    return header + "\n" + "\n".join(events) + "\n"


def _uses_fullscreen_black_matte(clip, canvas):
    transform = clip.get("transform") or {}
    return (
        clip.get("kind") in {"background", "overlay"}
        and clip.get("fitMode") == "contain"
        and transform.get("x") == 0
        and transform.get("y") == 0
        and transform.get("width") == canvas["width"]
        and transform.get("height") == canvas["height"]
    )


def _fit_filter(clip, canvas):
    transform = clip["transform"]
    width, height = round(transform["width"]), round(transform["height"])
    fit = clip.get("fitMode", "cover")
    if fit == "contain":
        matte_alpha = 1 if _uses_fullscreen_black_matte(clip, canvas) else 0
        return "scale=%d:%d:force_original_aspect_ratio=decrease,pad=%d:%d:(ow-iw)/2:(oh-ih)/2:color=black@%d" % (width, height, width, height, matte_alpha)
    if fit == "fit-width":
        return "scale=%d:-2,crop=%d:%d:(iw-ow)/2:(ih-oh)/2" % (width, width, height)
    if fit == "fit-height":
        return "scale=-2:%d,crop=%d:%d:(iw-ow)/2:(ih-oh)/2" % (height, width, height)
    if fit in {"stretch", "free"}:
        return "scale=%d:%d" % (width, height)
    return "scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d" % (width, height, width, height)


def _mask_filter(clip):
    mask = clip.get("mask", "rectangle")
    if mask == "circle":
        return ",format=rgba,geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='if(lte(hypot(X-W/2,Y-H/2),min(W,H)/2),255,0)'"
    if mask == "rounded-rectangle":
        radius = max(8, round(min(clip["transform"]["width"], clip["transform"]["height"]) * 0.08))
        return ",format=rgba,geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='if(lte(hypot(max(abs(X-W/2)-(W/2-%d),0),max(abs(Y-H/2)-(H/2-%d),0)),%d),255,0)'" % (radius, radius, radius)
    return ""


def build_ffmpeg_command(composition, asset_inputs, avatar_path, voice_path, captions_path, output_path):
    canvas = composition["canvas"]
    width, height, fps = canvas["width"], canvas["height"], canvas["fps"]
    duration_seconds = canvas["durationMs"] / 1000.0
    command = [
        "ffmpeg", "-y", "-f", "lavfi", "-i",
        "color=c=%s:s=%dx%d:r=%d:d=%.3f" % (canvas["background"], width, height, fps, duration_seconds),
        "-i", avatar_path,
        "-i", voice_path,
    ]
    referenced_ids = []
    for clip in composition["visualClips"] + composition.get("musicClips", []):
        asset_id = clip.get("assetId")
        if isinstance(asset_id, str) and asset_id not in referenced_ids:
            referenced_ids.append(asset_id)
    input_indices = {}
    next_index = 3
    for asset_id in referenced_ids:
        item = asset_inputs.get(asset_id)
        if not isinstance(item, dict) or item.get("kind") not in {"image", "video", "audio"}:
            raise CompositionValidationError("invalid downloaded asset")
        if item["kind"] == "image":
            command.extend(["-loop", "1", "-framerate", str(fps), "-i", item["path"]])
        elif item["kind"] == "video":
            command.extend(["-stream_loop", "-1", "-i", item["path"]])
        else:
            command.extend(["-stream_loop", "-1", "-i", item["path"]])
        input_indices[asset_id] = next_index
        next_index += 1

    filters = ["[0:v]format=rgba[base0]"]
    base_label = "base0"
    for index, clip in enumerate(composition["visualClips"]):
        source_index = 1 if clip["kind"] == "avatar" else input_indices[clip["assetId"]]
        start = clip["startMs"] / 1000.0
        end = (clip["startMs"] + clip["durationMs"]) / 1000.0
        offset = clip.get("sourceOffsetMs", 0) / 1000.0
        duration = clip["durationMs"] / 1000.0
        clip_label = "clip%d" % index
        chain = "[%d:v]trim=start=%.3f:duration=%.3f,setpts=PTS-STARTPTS+%.3f/TB,%s%s" % (
            source_index, offset, duration, start, _fit_filter(clip, canvas), _mask_filter(clip),
        )
        opacity = float(clip.get("opacity", 1))
        if opacity < 1:
            chain += ",format=rgba,colorchannelmixer=aa=%.3f" % opacity
        rotation = float(clip["transform"].get("rotation", 0))
        if abs(rotation) > 0.001:
            chain += ",rotate=%.6f*PI/180:c=none:ow=rotw(iw):oh=roth(ih)" % rotation
        filters.append(chain + "[%s]" % clip_label)
        output_label = "base%d" % (index + 1)
        transform = clip["transform"]
        filters.append("[%s][%s]overlay=%d:%d:eof_action=pass:shortest=0:enable='between(t,%.3f,%.3f)'[%s]" % (
            base_label, clip_label, round(transform["x"]), round(transform["y"]), start, end, output_label,
        ))
        base_label = output_label
    if captions_path:
        subtitle_path = captions_path.replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
        filters.append("[%s]subtitles=filename='%s'[videoout]" % (base_label, subtitle_path))
    else:
        filters.append("[%s]format=yuv420p[videoout]" % base_label)

    music_labels = []
    for index, clip in enumerate(composition.get("musicClips", [])):
        source_index = input_indices[clip["assetId"]]
        start = clip["startMs"]
        duration = clip["durationMs"] / 1000.0
        offset = clip.get("sourceOffsetMs", 0) / 1000.0
        label = "music%d" % index
        filters.append("[%d:a]atrim=start=%.3f:duration=%.3f,asetpts=PTS-STARTPTS,volume=%.3f,adelay=%d|%d[%s]" % (
            source_index, offset, duration, float(clip.get("volume", 1)), start, start, label,
        ))
        music_labels.append(label)
    if music_labels:
        filters.append("[2:a]atrim=duration=%.3f,asetpts=PTS-STARTPTS[voice]" % duration_seconds)
        filters.append("[voice]%samix=inputs=%d:duration=first:normalize=0[audioout]" % (
            "".join("[%s]" % label for label in music_labels), len(music_labels) + 1,
        ))

    command.extend(["-filter_complex", ";".join(filters), "-map", "[videoout]"])
    command.extend(["-map", "[audioout]" if music_labels else "2:a"])
    command.extend([
        "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-profile:v", "high",
        "-pix_fmt", "yuv420p", "-r", str(fps), "-c:a", "aac", "-b:a", "192k",
        "-t", "%.3f" % duration_seconds, "-movflags", "+faststart", output_path,
    ])
    return command
