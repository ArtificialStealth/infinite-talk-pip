"""
InfiniteTalk + PIP Composition Handler
======================================

Production-ready handler for generating talking avatar videos with PIP composition.

API Input Schema:
{
    "input_type": "image",          // "image" or "video"
    "person_count": "single",       // "single" or "multi"
    
    // Avatar input (one of these required)
    "image_url": "https://...",     // URL to avatar image
    "image_base64": "...",          // OR base64 encoded image
    "image_path": "/path/...",      // OR local path
    
    // Audio input (one of these required)
    "wav_url": "https://...",       // URL to audio file
    "wav_base64": "...",            // OR base64 encoded audio
    "wav_path": "/path/...",        // OR local path
    
    // Generation settings
    "prompt": "A person talking naturally",
    "width": 256,                   // Avatar width (256 recommended for PIP)
    "height": 384,                  // Avatar height (384 recommended for PIP)
    "max_frame": null,              // Auto-calculated from audio if null
    
    // PIP Mode settings
    "pip_mode": true,               // Enable PIP composition
    "background_images": [          // 1-10 product image URLs
        "https://...",
        "https://..."
    ],
    
    // Output settings
    "network_volume": false         // If true, save to /runpod-volume/
}

API Output Schema:
{
    "video": "base64...",           // Base64 encoded video (if network_volume=false)
    "video_path": "/runpod-volume/...",  // Path (if network_volume=true)
    "duration": 12.5,               // Video duration in seconds
    "resolution": "1080x1920",      // Output resolution (PIP mode)
    "pip_mode": true                // Whether PIP was applied
}

Error Response:
{
    "error": "Error message",
    "error_code": "ERROR_CODE",
    "details": "Additional details"
}
"""

import runpod
import os
import websocket
import base64
import json
import uuid
import logging
import urllib.request
import urllib.parse
import binascii
import subprocess
import shutil
import time
import traceback

# Try to import librosa, fallback to ffprobe for audio duration
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    logging.warning("librosa not available, using ffprobe for audio duration")

# ============================================
# CONFIGURATION
# ============================================

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ComfyUI server
SERVER_ADDRESS = os.getenv("SERVER_ADDRESS", "127.0.0.1")
CLIENT_ID = str(uuid.uuid4())

# PIP Composition settings
PIP_CONFIG = {
    "final_width": 1080,
    "final_height": 1920,
    "fps": 25,
    "avatar_scale": 0.35,           # Avatar is 35% of final height
    "avatar_x": 0,                  # Flush with left edge
    "avatar_y_offset": 0,           # Flush with bottom edge
    "chromakey_color": "0x35D346",  # InfiniteTalk green screen color
    "chromakey_similarity": 0.15,   # How much green to remove
    "chromakey_blend": 0.05,        # Edge blending
    "max_background_images": 10,    # Maximum background images
    "download_timeout": 60,         # Download timeout in seconds
}

# Error codes
class ErrorCode:
    INVALID_INPUT = "INVALID_INPUT"
    DOWNLOAD_FAILED = "DOWNLOAD_FAILED"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    COMFYUI_ERROR = "COMFYUI_ERROR"
    GENERATION_FAILED = "GENERATION_FAILED"
    PIP_COMPOSITION_FAILED = "PIP_COMPOSITION_FAILED"
    ENCODING_FAILED = "ENCODING_FAILED"
    TIMEOUT = "TIMEOUT"


# ============================================
# UTILITY FUNCTIONS
# ============================================

def truncate_for_log(text, max_length=100):
    """Truncate text for logging"""
    if not text:
        return "None"
    text = str(text)
    if len(text) <= max_length:
        return text
    return f"{text[:max_length]}... ({len(text)} chars total)"


def create_error_response(message, error_code, details=None):
    """Create standardized error response"""
    response = {
        "error": message,
        "error_code": error_code,
    }
    if details:
        response["details"] = details
    logger.error(f"Error [{error_code}]: {message}")
    return response


def download_file(url, output_path, timeout=60):
    """Download file from URL with timeout and validation"""
    try:
        logger.info(f"📥 Downloading: {truncate_for_log(url)}")
        
        # Use wget for more reliable downloads
        result = subprocess.run(
            ["wget", "-O", output_path, "--no-verbose", f"--timeout={timeout}", url],
            capture_output=True,
            text=True,
            timeout=timeout + 10,
        )

        if result.returncode != 0:
            raise Exception(f"wget failed: {result.stderr}")
        
        if not os.path.exists(output_path):
            raise Exception("File not created after download")
        
        file_size = os.path.getsize(output_path)
        if file_size == 0:
            os.remove(output_path)
            raise Exception("Downloaded file is empty")
        
        logger.info(f"✅ Downloaded: {file_size} bytes -> {output_path}")
        return output_path
        
    except subprocess.TimeoutExpired:
        raise Exception(f"Download timeout after {timeout}s")
    except Exception as e:
        raise Exception(f"Download failed: {str(e)}")


def save_base64_to_file(base64_data, output_path):
    """Decode and save base64 data to file"""
    try:
        # Handle data URL prefix
        if "," in base64_data:
            base64_data = base64_data.split(",")[1]
        
        decoded = base64.b64decode(base64_data)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(decoded)
        
        logger.info(f"✅ Saved base64 to: {output_path} ({len(decoded)} bytes)")
        return output_path
        
    except Exception as e:
        raise Exception(f"Base64 decode failed: {str(e)}")


def process_input_file(job_input, task_dir, file_prefix, extensions):
    """
    Process file input from path, URL, or base64.
    Returns file path or raises exception.
    """
    path_key = f"{file_prefix}_path"
    url_key = f"{file_prefix}_url"
    base64_key = f"{file_prefix}_base64"
    
    output_filename = f"input_{file_prefix}{extensions[0]}"
    output_path = os.path.join(task_dir, output_filename)
    
    if path_key in job_input and job_input[path_key]:
        # Direct path
        path = job_input[path_key]
        if not os.path.exists(path):
            raise Exception(f"File not found: {path}")
        return path
        
    elif url_key in job_input and job_input[url_key]:
        # Download from URL
        return download_file(
            job_input[url_key], 
            output_path,
            timeout=PIP_CONFIG["download_timeout"]
        )
        
    elif base64_key in job_input and job_input[base64_key]:
        # Decode base64
        return save_base64_to_file(job_input[base64_key], output_path)
    
    return None


def get_audio_duration(audio_path):
    """Get audio duration using librosa or ffprobe"""
    try:
        if HAS_LIBROSA:
            return librosa.get_duration(path=audio_path)
        else:
            # Fallback to ffprobe
            cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                audio_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return float(result.stdout.strip())
    except Exception as e:
        logger.warning(f"Could not get audio duration: {e}")
        return None


def get_video_duration(video_path):
    """Get video duration using ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except Exception as e:
        logger.warning(f"Could not get video duration: {e}")
        return 10.0


def calculate_max_frames(wav_path, wav_path_2=None, fps=25):
    """Calculate max frames from audio duration"""
    durations = []
    
    duration1 = get_audio_duration(wav_path)
    if duration1:
        durations.append(duration1)
        logger.info(f"Audio 1 duration: {duration1:.2f}s")
    
    if wav_path_2:
        duration2 = get_audio_duration(wav_path_2)
        if duration2:
            durations.append(duration2)
            logger.info(f"Audio 2 duration: {duration2:.2f}s")
    
    if not durations:
        logger.warning("Could not calculate audio duration, using default 81 frames")
        return 81
    
    max_duration = max(durations)
    max_frames = int(max_duration * fps) + 81  # Extra frames for safety
    logger.info(f"Max duration: {max_duration:.2f}s -> {max_frames} frames")
    return max_frames


def cleanup_temp_dir(task_dir):
    """Clean up temporary directory"""
    try:
        if os.path.exists(task_dir):
            shutil.rmtree(task_dir)
            logger.info(f"🧹 Cleaned up: {task_dir}")
    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")


# ============================================
# COMFYUI FUNCTIONS
# ============================================

def load_workflow(workflow_path):
    """Load workflow JSON file"""
    with open(workflow_path, "r") as f:
        return json.load(f)


def get_workflow_path(input_type, person_count):
    """Get appropriate workflow file"""
    if input_type == "image":
        return "/I2V_single.json" if person_count == "single" else "/I2V_multi.json"
    else:
        return "/V2V_single.json" if person_count == "single" else "/V2V_multi.json"


def queue_prompt(prompt, client_id):
    """Queue prompt to ComfyUI"""
    url = f"http://{SERVER_ADDRESS}:8188/prompt"
    data = json.dumps({"prompt": prompt, "client_id": client_id}).encode("utf-8")
    
    req = urllib.request.Request(url, data=data)
    req.add_header("Content-Type", "application/json")
    
    response = urllib.request.urlopen(req, timeout=30)
    return json.loads(response.read())


def get_history(prompt_id):
    """Get execution history from ComfyUI"""
    url = f"http://{SERVER_ADDRESS}:8188/history/{prompt_id}"
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read())


def wait_for_comfyui(max_wait=180):
    """Wait for ComfyUI server to be ready"""
    http_url = f"http://{SERVER_ADDRESS}:8188/"
    logger.info(f"Waiting for ComfyUI at {http_url}...")
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            urllib.request.urlopen(http_url, timeout=5)
            logger.info("✅ ComfyUI is ready")
            return True
        except Exception:
            time.sleep(1)
    
    raise Exception(f"ComfyUI not ready after {max_wait}s")


def run_workflow(prompt, client_id, input_type, person_count):
    """Run workflow and wait for completion"""
    # Connect WebSocket
    ws_url = f"ws://{SERVER_ADDRESS}:8188/ws?clientId={client_id}"
    logger.info(f"Connecting to WebSocket: {ws_url}")
    
    ws = websocket.WebSocket()
    ws.settimeout(600)  # 10 minute timeout
    
    max_attempts = 36  # 3 minutes
    for attempt in range(max_attempts):
        try:
            ws.connect(ws_url)
            logger.info("✅ WebSocket connected")
            break
        except Exception as e:
            if attempt == max_attempts - 1:
                raise Exception(f"WebSocket connection failed: {e}")
            time.sleep(5)
    
    try:
        # Queue prompt
        result = queue_prompt(prompt, client_id)
        prompt_id = result["prompt_id"]
        logger.info(f"Workflow started: {prompt_id}")
        
        # Wait for completion
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message["type"] == "executing":
                    data = message["data"]
                    if data["node"]:
                        logger.info(f"Executing node: {data['node']}")
                    if data["node"] is None and data["prompt_id"] == prompt_id:
                        logger.info("✅ Workflow completed")
                        break
        
        # Get results
        history = get_history(prompt_id)[prompt_id]
        
        output_videos = []
        for node_id, node_output in history["outputs"].items():
            if "gifs" in node_output:
                for video in node_output["gifs"]:
                    video_path = video.get("fullpath")
                    if video_path and os.path.exists(video_path):
                        output_videos.append(video_path)
                        logger.info(f"Found output video: {video_path}")
        
        return output_videos
        
    finally:
        ws.close()
        logger.info("WebSocket closed")


# ============================================
# PIP COMPOSITION FUNCTIONS
# ============================================

def download_background_images(image_urls, task_dir):
    """Download background images for slideshow"""
    paths = []
    max_images = PIP_CONFIG["max_background_images"]
    
    for i, url in enumerate(image_urls[:max_images]):
        try:
            output_path = os.path.join(task_dir, f"bg_{i}.jpg")
            download_file(url, output_path, timeout=30)
            paths.append(output_path)
        except Exception as e:
            logger.warning(f"Failed to download background {i}: {e}")
    
    logger.info(f"Downloaded {len(paths)}/{len(image_urls)} background images")
    return paths


def create_background_slideshow(image_paths, duration, output_path):
    """Create background slideshow from images"""
    config = PIP_CONFIG
    
    if not image_paths:
        # Create solid color background
        logger.info("No images, creating solid background")
        cmd = [
            'ffmpeg', '-y', '-f', 'lavfi',
            '-i', f'color=c=#1a1a2e:s={config["final_width"]}x{config["final_height"]}:r={config["fps"]}:d={duration}',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            output_path
        ]
        subprocess.run(cmd, capture_output=True, timeout=60)
        return os.path.exists(output_path)
    
    time_per_image = duration / len(image_paths)
    logger.info(f"Creating slideshow: {len(image_paths)} images, {time_per_image:.2f}s each")
    
    # Build FFmpeg command
    input_args = []
    filter_parts = []
    
    for i, path in enumerate(image_paths):
        input_args.extend(['-loop', '1', '-t', str(time_per_image), '-i', path])
        # Scale to fit (not crop) - adds black bars if aspect ratio differs
        # Use pad to center the image on the 9:16 canvas
        filter_parts.append(
            f"[{i}:v]scale={config['final_width']}:{config['final_height']}:"
            f"force_original_aspect_ratio=decrease,"
            f"pad={config['final_width']}:{config['final_height']}:(ow-iw)/2:(oh-ih)/2:black,"
            f"setsar=1[v{i}]"
        )
    
    concat = ''.join([f"[v{i}]" for i in range(len(image_paths))])
    filter_parts.append(f"{concat}concat=n={len(image_paths)}:v=1:a=0[outv]")
    
    cmd = [
        'ffmpeg', '-y', *input_args,
        '-filter_complex', ';'.join(filter_parts),
        '-map', '[outv]',
        '-c:v', 'libx264', '-preset', 'fast', '-pix_fmt', 'yuv420p',
        '-t', str(duration),
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    
    if result.returncode != 0:
        logger.error(f"Slideshow creation failed: {result.stderr[:500]}")
        # Fallback: use first image as static background (fit, not crop)
        cmd = [
            'ffmpeg', '-y', '-loop', '1', '-i', image_paths[0],
            '-vf', f'scale={config["final_width"]}:{config["final_height"]}:'
                   f'force_original_aspect_ratio=decrease,'
                   f'pad={config["final_width"]}:{config["final_height"]}:(ow-iw)/2:(oh-ih)/2:black',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-t', str(duration),
            output_path
        ]
        subprocess.run(cmd, capture_output=True, timeout=60)
    
    return os.path.exists(output_path)


def composite_pip_video(background_path, avatar_path, audio_path, output_path):
    """Composite avatar over background with chromakey"""
    config = PIP_CONFIG
    
    # Calculate avatar dimensions and position
    avatar_height = int(config["final_height"] * config["avatar_scale"])
    avatar_width = int(avatar_height * 2 / 3)  # 2:3 aspect ratio
    avatar_x = config["avatar_x"]
    avatar_y = config["final_height"] - avatar_height - config["avatar_y_offset"]
    
    logger.info(f"🎬 Compositing: avatar {avatar_width}x{avatar_height} at ({avatar_x}, {avatar_y})")
    
    # FFmpeg filter: chromakey + scale + overlay
    filter_complex = (
        f"[1:v]chromakey={config['chromakey_color']}:"
        f"{config['chromakey_similarity']}:{config['chromakey_blend']},"
        f"scale={avatar_width}:{avatar_height}[avatar];"
        f"[0:v][avatar]overlay={avatar_x}:{avatar_y}[outv]"
    )
    
    cmd = [
        'ffmpeg', '-y',
        '-i', background_path,
        '-i', avatar_path,
        '-i', audio_path,
        '-filter_complex', filter_complex,
        '-map', '[outv]',
        '-map', '2:a',
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
        '-c:a', 'aac', '-b:a', '192k',
        '-pix_fmt', 'yuv420p',
        '-shortest',
        '-movflags', '+faststart',
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    
    if result.returncode != 0:
        logger.error(f"Composition failed: {result.stderr[:500]}")
        return False
    
    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        logger.info(f"✅ PIP video created: {size} bytes")
        return True
    
    return False


# ============================================
# MAIN HANDLER
# ============================================

def handler(job):
    """Main handler for RunPod serverless"""
    job_input = job.get("input", {})
    job_id = job.get("id", str(uuid.uuid4()))
    task_id = f"task_{uuid.uuid4()}"
    task_dir = f"/{task_id}"
    
    logger.info("=" * 60)
    logger.info(f"🚀 JOB STARTED: {job_id}")
    logger.info("=" * 60)
    
    # Log sanitized input
    log_input = {k: truncate_for_log(v) for k, v in job_input.items()}
    logger.info(f"Input: {json.dumps(log_input, indent=2)}")
    
    try:
        os.makedirs(task_dir, exist_ok=True)
        
        # ============================================
        # VALIDATE INPUTS
        # ============================================
        
        input_type = job_input.get("input_type", "image")
        person_count = job_input.get("person_count", "single")
        pip_mode = job_input.get("pip_mode", False)
        background_images_urls = job_input.get("background_images", [])
        
        if input_type not in ["image", "video"]:
            return create_error_response(
                f"Invalid input_type: {input_type}",
                ErrorCode.INVALID_INPUT,
                "input_type must be 'image' or 'video'"
            )
        
        if person_count not in ["single", "multi"]:
            return create_error_response(
                f"Invalid person_count: {person_count}",
                ErrorCode.INVALID_INPUT,
                "person_count must be 'single' or 'multi'"
            )
        
        # ============================================
        # PROCESS INPUT FILES
        # ============================================
        
        # Process media input (image or video)
        media_prefix = "image" if input_type == "image" else "video"
        media_ext = [".jpg", ".png"] if input_type == "image" else [".mp4", ".mov"]
        
        try:
            media_path = process_input_file(job_input, task_dir, media_prefix, media_ext)
            if not media_path:
                # Use default
                media_path = "/examples/image.jpg"
                logger.info(f"Using default: {media_path}")
        except Exception as e:
            return create_error_response(
                f"Failed to process {media_prefix} input",
                ErrorCode.DOWNLOAD_FAILED,
                str(e)
            )
        
        # Process audio input
        try:
            wav_path = process_input_file(job_input, task_dir, "wav", [".wav", ".mp3"])
            if not wav_path:
                wav_path = "/examples/audio.mp3"
                logger.info(f"Using default audio: {wav_path}")
        except Exception as e:
            return create_error_response(
                "Failed to process audio input",
                ErrorCode.DOWNLOAD_FAILED,
                str(e)
            )
        
        # Second audio for multi-person
        wav_path_2 = None
        if person_count == "multi":
            try:
                wav_path_2 = process_input_file(
                    {k.replace("_2", ""): v for k, v in job_input.items() if "_2" in k},
                    task_dir, "wav2", [".wav", ".mp3"]
                )
            except Exception:
                pass
            if not wav_path_2:
                wav_path_2 = wav_path
        
        # Verify files exist
        if not os.path.exists(media_path):
            return create_error_response(
                f"Media file not found: {media_path}",
                ErrorCode.FILE_NOT_FOUND
            )
        
        if not os.path.exists(wav_path):
            return create_error_response(
                f"Audio file not found: {wav_path}",
                ErrorCode.FILE_NOT_FOUND
            )
        
        logger.info(f"Media: {media_path} ({os.path.getsize(media_path)} bytes)")
        logger.info(f"Audio: {wav_path} ({os.path.getsize(wav_path)} bytes)")
        
        # ============================================
        # CONFIGURE WORKFLOW
        # ============================================
        
        workflow_path = get_workflow_path(input_type, person_count)
        logger.info(f"Workflow: {workflow_path}")
        
        prompt = load_workflow(workflow_path)
        
        # Set workflow parameters
        prompt_text = job_input.get("prompt", "A person talking naturally")
        width = job_input.get("width", 256 if pip_mode else 512)
        height = job_input.get("height", 384 if pip_mode else 512)
        
        max_frame = job_input.get("max_frame")
        if not max_frame:
            max_frame = calculate_max_frames(wav_path, wav_path_2)
        
        # Configure workflow nodes
        if input_type == "image":
            prompt["284"]["inputs"]["image"] = media_path
        else:
            prompt["228"]["inputs"]["video"] = media_path
        
        prompt["125"]["inputs"]["audio"] = wav_path
        prompt["241"]["inputs"]["positive_prompt"] = prompt_text
        prompt["245"]["inputs"]["value"] = width
        prompt["246"]["inputs"]["value"] = height
        prompt["270"]["inputs"]["value"] = max_frame
        
        if person_count == "multi" and wav_path_2:
            if "307" in prompt:
                prompt["307"]["inputs"]["audio"] = wav_path_2
            elif "313" in prompt:
                prompt["313"]["inputs"]["audio"] = wav_path_2
        
        logger.info(f"Settings: {width}x{height}, {max_frame} frames, prompt='{prompt_text[:50]}...'")
        
        # ============================================
        # GENERATE AVATAR VIDEO
        # ============================================
        
        logger.info("=" * 40)
        logger.info("🎭 GENERATING AVATAR VIDEO...")
        logger.info("=" * 40)
        
        wait_for_comfyui()
        
        output_videos = run_workflow(prompt, CLIENT_ID, input_type, person_count)
        
        if not output_videos:
            return create_error_response(
                "No video generated by InfiniteTalk",
                ErrorCode.GENERATION_FAILED
            )
        
        avatar_video_path = output_videos[0]
        avatar_duration = get_video_duration(avatar_video_path)
        logger.info(f"✅ Avatar video: {avatar_video_path} ({avatar_duration:.2f}s)")
        
        # ============================================
        # PIP COMPOSITION (if enabled)
        # ============================================
        
        final_video_path = avatar_video_path
        final_resolution = f"{width}x{height}"
        pip_applied = False
        
        if pip_mode:
            logger.info("=" * 40)
            logger.info("🎬 PIP COMPOSITION...")
            logger.info("=" * 40)
            
            pip_dir = os.path.join(task_dir, "pip")
            os.makedirs(pip_dir, exist_ok=True)
            
            # Download background images
            bg_paths = download_background_images(background_images_urls, pip_dir)
            
            # Create background slideshow
            background_path = os.path.join(pip_dir, "background.mp4")
            if not create_background_slideshow(bg_paths, avatar_duration, background_path):
                return create_error_response(
                    "Failed to create background slideshow",
                    ErrorCode.PIP_COMPOSITION_FAILED
                )
            
            # Composite final video
            final_path = os.path.join(pip_dir, "final_pip.mp4")
            if not composite_pip_video(background_path, avatar_video_path, wav_path, final_path):
                return create_error_response(
                    "Failed to composite PIP video",
                    ErrorCode.PIP_COMPOSITION_FAILED
                )
            
            final_video_path = final_path
            final_resolution = f"{PIP_CONFIG['final_width']}x{PIP_CONFIG['final_height']}"
            pip_applied = True
            logger.info("✅ PIP composition complete")
        
        # ============================================
        # RETURN VIDEO
        # ============================================
        
        final_duration = get_video_duration(final_video_path)
        final_size = os.path.getsize(final_video_path)
        
        logger.info("=" * 40)
        logger.info("📦 PREPARING OUTPUT...")
        logger.info(f"Video: {final_video_path}")
        logger.info(f"Size: {final_size} bytes")
        logger.info(f"Duration: {final_duration:.2f}s")
        logger.info(f"Resolution: {final_resolution}")
        logger.info("=" * 40)
        
        use_network_volume = job_input.get("network_volume", False)
        
        if use_network_volume:
            # Copy to network volume
            output_filename = f"infinitetalk_{task_id}.mp4"
            volume_path = f"/runpod-volume/{output_filename}"
            
            try:
                shutil.copy2(final_video_path, volume_path)
                logger.info(f"✅ Saved to network volume: {volume_path}")
                
                return {
                    "video_path": volume_path,
                    "duration": final_duration,
                    "resolution": final_resolution,
                    "pip_mode": pip_applied,
                    "size_bytes": final_size,
                }
            except Exception as e:
                return create_error_response(
                    "Failed to save to network volume",
                    ErrorCode.ENCODING_FAILED,
                    str(e)
                )
        else:
            # Return as base64
            try:
                with open(final_video_path, "rb") as f:
                    video_base64 = base64.b64encode(f.read()).decode("utf-8")
                
                logger.info(f"✅ Encoded to base64: {len(video_base64)} chars")
                
                return {
                    "video": video_base64,
                    "duration": final_duration,
                    "resolution": final_resolution,
                    "pip_mode": pip_applied,
                    "size_bytes": final_size,
                }
            except Exception as e:
                return create_error_response(
                    "Failed to encode video to base64",
                    ErrorCode.ENCODING_FAILED,
                    str(e)
                )
    
    except Exception as e:
        logger.error(f"❌ Unhandled error: {e}")
        logger.error(traceback.format_exc())
        return create_error_response(
            str(e),
            ErrorCode.GENERATION_FAILED,
            traceback.format_exc()
        )
    
    finally:
        # Cleanup
        cleanup_temp_dir(task_dir)
        logger.info("=" * 60)
        logger.info(f"🏁 JOB COMPLETED: {job_id}")
        logger.info("=" * 60)


# Start RunPod serverless
runpod.serverless.start({"handler": handler})
