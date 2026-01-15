# InfiniteTalk + PIP Composition

This is a modified version of InfiniteTalk that includes **PIP (Picture-in-Picture) composition**.

## What It Does

1. **Generates a talking avatar** using InfiniteTalk (green screen background)
2. **Creates a background slideshow** from your product images
3. **Removes the green screen** using FFmpeg chromakey
4. **Composites the avatar** in the bottom-left corner over the slideshow
5. **Outputs a final 1080x1920 video** ready for TikTok/Reels/Shorts

## Input Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image_url` | string | Yes | - | Avatar image URL (green screen) |
| `wav_url` | string | Yes | - | Audio/TTS file URL |
| `pip_mode` | boolean | No | false | Enable PIP composition |
| `background_images` | array | No | [] | Product image URLs for slideshow |
| `width` | int | No | 256 | Avatar generation width |
| `height` | int | No | 384 | Avatar generation height |
| `prompt` | string | No | "A person talking naturally" | Generation prompt |

### Example Input

```json
{
  "input": {
    "input_type": "image",
    "person_count": "single",
    "image_url": "https://your-bucket/avatar.png",
    "wav_url": "https://your-bucket/audio.mp3",
    "width": 256,
    "height": 384,
    "pip_mode": true,
    "background_images": [
      "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=800",
      "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=800",
      "https://images.unsplash.com/photo-1560769629-975ec94e6a86?w=800"
    ]
  }
}
```

### Output

```json
{
  "video": "base64-encoded-video..."
}
```

Or with network volume:

```json
{
  "video_path": "/runpod-volume/infinitetalk_task_xxx.mp4"
}
```

## Deployment to RunPod

### Option 1: Build Locally and Push

```bash
cd Infinitetalk_Runpod_hub-main

# Build with PIP Dockerfile
docker build -f Dockerfile.pip -t your-dockerhub/infinitetalk-pip:v1 .

# Push to Docker Hub
docker push your-dockerhub/infinitetalk-pip:v1
```

### Option 2: Fork on GitHub and Use RunPod's Builder

1. Fork the repo to your GitHub account
2. Replace `handler.py` with `handler_pip.py` contents
3. Modify the Dockerfile to install ffmpeg:
   ```dockerfile
   RUN apt-get update && apt-get install -y wget ffmpeg && rm -rf /var/lib/apt/lists/*
   ```
4. In RunPod, create a new serverless endpoint with:
   - **GitHub repo**: Your forked repo
   - **Docker file path**: `Dockerfile`

### Create RunPod Endpoint

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click "New Endpoint"
3. Settings:
   - **Name**: infinitetalk-pip
   - **Docker Image**: `your-dockerhub/infinitetalk-pip:v1`
   - **GPU**: Select "32 GB PRO" (L40S/RTX 5090)
   - **Active Workers**: 0 (scale to zero)
   - **Max Workers**: 2
   - **Idle Timeout**: 5 seconds
4. Copy the endpoint ID

### Update Your .env

```bash
RUNPOD_PIP_ENDPOINT_ID=your-pip-endpoint-id
```

## PIP Composition Settings

The composition uses these settings (modify in `handler_pip.py`):

```python
FINAL_WIDTH = 1080       # Output video width
FINAL_HEIGHT = 1920      # Output video height (9:16)
FPS = 25                 # Frames per second
AVATAR_SCALE = 0.35      # Avatar size (35% of height)
CHROMAKEY_COLOR = "0x35D346"  # Green screen color
CHROMAKEY_SIMILARITY = 0.15   # How much green to remove
CHROMAKEY_BLEND = 0.05        # Edge blending
```

## Cost Estimate

| Step | Duration | Cost |
|------|----------|------|
| Avatar generation | ~2-3 min | ~$0.05-0.10 |
| PIP composition | ~10-20 sec | Included (CPU) |
| **Total** | ~2.5-3.5 min | **~$0.05-0.10** |

## Troubleshooting

### Green screen not removed properly

The chromakey is tuned for InfiniteTalk's specific green (#35D346). If your avatars have different green:

1. Extract a frame from a generated video
2. Sample the green color
3. Update `CHROMAKEY_COLOR` in `handler_pip.py`

### Avatar cutoff at edges

Increase margins in `composite_pip_video()`:
```python
avatar_x = 10  # Add left margin
avatar_y = FINAL_HEIGHT - avatar_height - 10  # Add bottom margin
```

### Slideshow images not showing

- Verify all image URLs are publicly accessible
- Check logs for download errors
- Max 10 images are supported

## Integration Example

```typescript
// From your Next.js API
const response = await fetch(`https://api.runpod.ai/v2/${ENDPOINT_ID}/run`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${RUNPOD_API_KEY}`,
  },
  body: JSON.stringify({
    input: {
      input_type: 'image',
      person_count: 'single',
      image_url: avatarUrl,
      wav_url: audioUrl,
      width: 256,
      height: 384,
      pip_mode: true,
      background_images: productImageUrls,
    },
  }),
});
```
