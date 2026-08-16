# Dynamic UGC Compositor Implementation Plan

> **For agentic workers:** Execute inline with test-first checkpoints.

**Goal:** Render exact Fireship Studio compositions on the existing InfiniteTalk RunPod endpoint.

**Architecture:** A pure Python composition module validates editor documents and builds bounded FFmpeg inputs, filters, ASS captions, and output metadata. The existing handler invokes it after avatar generation for `ugc_composition_v1` jobs; the web pipeline submits and reconciles that combined job.

**Tech Stack:** Python 3, FFmpeg/ffprobe, unittest, TypeScript, Vitest, RunPod, Supabase.

---

### Task 1: Build the tested composition module

- [ ] Add failing tests for document validation, visual z-order/timing/transform filters, cover/contain fit, and styled caption ASS output.
- [ ] Implement `composition_renderer.py` with strict bounded validation and shell-free FFmpeg command construction.
- [ ] Run `python3 -m unittest -v test_composition_renderer.py` and commit.

### Task 2: Add the combined RunPod job

- [ ] Add handler contract tests for `render_type=ugc_composition_v1` dispatch.
- [ ] Download referenced assets, render after InfiniteTalk avatar generation, and return base64 video plus exact resolution/duration metadata.
- [ ] Run Python tests, compile all Python modules, and commit.

### Task 3: Integrate the web pipeline

- [ ] Add failing TypeScript tests for the combined composition payload, reconciliation, endpoint fallback, and submission-failure project restoration.
- [ ] Implement the adapter/pipeline/runtime and failure-state changes.
- [ ] Run focused tests, the full web suite, and the production build; commit and deploy.

### Task 4: Production proof

- [ ] Push the worker `main` branch and wait for RunPod deployment.
- [ ] Reset only the already-released failed test project to `requested` after verifying reservation state.
- [ ] Submit one retry, monitor to completion, and verify the stored MP4's bytes, dimensions, duration, and authenticated playback.
