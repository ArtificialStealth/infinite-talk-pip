# Dynamic UGC Compositor Design

## Goal

Upgrade the existing InfiniteTalk PIP RunPod worker into the production renderer for Fireship Studio compositions without adding a second endpoint.

## Architecture

The web app submits one combined job to `RUNPOD_UGC_PIP_ENDPOINT_ID`. InfiniteTalk first generates the lip-synced avatar from the selected avatar image and exact preview voice. The worker then passes that avatar video, downloaded user assets, the immutable composition document, and voice-derived captions to a focused FFmpeg compositor.

The compositor validates schema version, canvas dimensions, duration, clip count, clip timing, transforms, opacity, z-order, and asset references before running FFmpeg. Visual clips are layered in z-order and enabled only for their timeline interval. Images loop for the clip duration; videos loop and honor source offsets. Fit modes map to deterministic scale/crop/pad filters. Rectangle, rounded-rectangle, square, and circle masks are supported. The saved voice is the master audio track.

Captions are rendered from alignment phrases and words into an ASS subtitle stream. Position, font size/weight, colors, outline, background opacity, alignment, uppercase behavior, and active-word highlighting come from `captionStyle`. Output is H.264 High Profile/YUV420p with AAC audio, constant 30 fps, CRF 18, and fast-start metadata at the exact editor canvas size.

## Integration

`RunPodPipVideoRenderer` gains a combined composition submission method. `ProviderUgcRenderPipeline.startComposition` prefers it and reconciles the resulting base64 or URL output using the existing verified-video storage boundary. The separate animator/compositor path remains available when dedicated endpoint variables are configured.

The production runtime falls back to the existing PIP endpoint for avatar and composition roles. A failed submission marks the render failed, releases the credit reservation, and restores the project to `requested` with reservation fields cleared so a safe retry is possible.

## Safety and verification

Downloads remain HTTPS-only signed URLs supplied by Fireship. Inputs are bounded by clip count, asset count, duration, and canvas dimensions. FFmpeg is invoked as an argument array, never through a shell. Tests cover validation, transform/fit/timing translation, z-order, captions, combined RunPod payloads, reconciliation, and failed-reservation restoration. The worker is pushed to `main`, the web app is deployed, and one real project is rendered and checked for MP4 type, nonzero bytes, portrait dimensions, duration, and playback.
