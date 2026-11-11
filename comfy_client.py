"""Minimal ComfyUI REST client for JARVIS.

Two generation paths:
  - generate_image(): uses a built-in SD1.5/SDXL graph. Works with zero
    configuration beyond a checkpoint file.
  - generate_video(): uses a user-supplied API-format workflow JSON
    (exported from ComfyUI via File → Export (API)). Video graphs vary
    wildly between AnimateDiff / SVD / Wan setups, so hardcoding one
    would break for most users; instead we patch the prompt and seed
    into whatever graph the user actually built and verified.

Uses only stdlib (urllib, json, uuid, time, os). No websocket-client
dependency — completion is detected by polling /history/{prompt_id},
which is slower by ~1s but has no install footprint.

ComfyUI API reference:
  POST /prompt              queue a workflow
  GET  /history/{id}        poll for completion + output filenames
  GET  /view?filename=...   download a generated file
"""

import json
import os
import time
import uuid
import urllib.request
import urllib.error
import urllib.parse
import logging

logger = logging.getLogger(__name__)

DEFAULT_SERVER = "http://127.0.0.1:8188"
DEFAULT_IMAGE_TIMEOUT = 180
DEFAULT_VIDEO_TIMEOUT = 900
POLL_INTERVAL = 1.0


class ComfyError(Exception):
    """Raised when ComfyUI is unreachable or a generation fails."""


class ComfyClient:
    def __init__(self, server_url=DEFAULT_SERVER, output_dir=None):
        self.server_url = (server_url or DEFAULT_SERVER).rstrip('/')
        self.output_dir = output_dir
        self.client_id = str(uuid.uuid4())
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

    # ----------------------------------------------------------
    # Health
    # ----------------------------------------------------------
    def is_available(self, timeout=3):
        """True if ComfyUI answers /system_stats. Cheap preflight."""
        try:
            with urllib.request.urlopen(
                self.server_url + "/system_stats", timeout=timeout
            ) as r:
                return r.status == 200
        except Exception:
            return False

    def system_stats(self):
        """Return ComfyUI's /system_stats payload (VRAM, device, versions)."""
        with urllib.request.urlopen(self.server_url + "/system_stats", timeout=5) as r:
            return json.loads(r.read().decode('utf-8'))

    # ----------------------------------------------------------
    # Low-level queue + poll + download
    # ----------------------------------------------------------
    def _queue(self, workflow):
        payload = {"prompt": workflow, "client_id": self.client_id}
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            self.server_url + "/prompt",
            data=data,
            headers={'Content-Type': 'application/json'}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                body = json.loads(r.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            detail = ''
            try:
                detail = e.read().decode('utf-8')[:800]
            except Exception:
                pass
            raise ComfyError(f"ComfyUI rejected the workflow (HTTP {e.code}): {detail}")
        except Exception as e:
            raise ComfyError(f"Could not reach ComfyUI at {self.server_url}: {e}")

        if body.get('node_errors'):
            raise ComfyError(f"Workflow validation failed: {json.dumps(body['node_errors'])[:800]}")
        prompt_id = body.get('prompt_id')
        if not prompt_id:
            raise ComfyError(f"ComfyUI returned no prompt_id: {body}")
        return prompt_id

    def _wait(self, prompt_id, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(
                    self.server_url + f"/history/{prompt_id}", timeout=10
                ) as r:
                    history = json.loads(r.read().decode('utf-8'))
            except Exception as e:
                raise ComfyError(f"Lost connection to ComfyUI while waiting: {e}")

            entry = history.get(prompt_id)
            if entry:
                status = entry.get('status', {})
                if status.get('status_str') == 'error':
                    msgs = status.get('messages', [])
                    raise ComfyError(f"ComfyUI generation failed: {json.dumps(msgs)[:800]}")
                if status.get('completed') or entry.get('outputs'):
                    return entry.get('outputs', {})
            time.sleep(POLL_INTERVAL)
        raise ComfyError(
            f"Generation timed out after {timeout}s. ComfyUI may still be working — "
            f"check its console, or raise the timeout."
        )

    def _download_outputs(self, outputs, prefix):
        """Download every image/video file referenced in outputs.
        Returns a list of dicts with 'path', 'filename', 'type'."""
        if not self.output_dir:
            raise ComfyError("No output directory configured")

        saved = []
        for node_id, node_out in outputs.items():
            for key in ('images', 'gifs', 'videos'):
                for item in node_out.get(key, []) or []:
                    filename = item.get('filename')
                    if not filename:
                        continue
                    params = urllib.parse.urlencode({
                        'filename': filename,
                        'subfolder': item.get('subfolder', ''),
                        'type': item.get('type', 'output')
                    })
                    try:
                        with urllib.request.urlopen(
                            self.server_url + "/view?" + params, timeout=60
                        ) as r:
                            blob = r.read()
                    except Exception as e:
                        logger.warning(f"Failed to download {filename}: {e}")
                        continue

                    ext = os.path.splitext(filename)[1] or '.bin'
                    safe_name = f"{prefix}_{node_id}_{filename}"
                    dest = os.path.join(self.output_dir, safe_name)
                    with open(dest, 'wb') as f:
                        f.write(blob)
                    saved.append({
                        'path': dest,
                        'filename': filename,
                        'type': 'video' if ext.lower() in ('.mp4', '.webm', '.gif', '.webp') else 'image',
                        'bytes': len(blob)
                    })
        if not saved:
            raise ComfyError("ComfyUI finished but produced no downloadable files")
        return saved

    # ----------------------------------------------------------
    # Workflow construction
    # ----------------------------------------------------------
    def _build_image_workflow(self, prompt, negative, checkpoint, width, height,
                              steps, cfg, seed):
        """Standard SD txt2img graph: loader → 2×CLIP encode → empty latent
        → KSampler → VAE decode → SaveImage. Node IDs are arbitrary strings;
        ComfyUI only cares about the links between them."""
        return {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": cfg, "denoise": 1.0,
                    "latent_image": ["5", 0],
                    "model": ["4", 0],
                    "negative": ["7", 0],
                    "positive": ["6", 0],
                    "sampler_name": "euler", "scheduler": "normal",
                    "seed": seed, "steps": steps
                }
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": checkpoint}
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {"batch_size": 1, "height": height, "width": width}
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["4", 1], "text": prompt}
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["4", 1], "text": negative}
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["3", 0], "vae": ["4", 2]}
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "jarvis", "images": ["8", 0]}
            }
        }

    @staticmethod
    def _patch_video_workflow(workflow, prompt, seed):
        """Walk a user-supplied API-format workflow and overwrite:
          - the first CLIPTextEncode node's text  (→ positive prompt)
          - every KSampler-ish node's seed       (→ reproducibility)
        Returns (workflow, patched_count). Raises if no prompt node found."""
        prompt_patched = 0
        seeds_patched = 0

        for node in workflow.values():
            if not isinstance(node, dict):
                continue
            ctype = node.get('class_type', '')
            inputs = node.setdefault('inputs', {})

            if ctype in ('CLIPTextEncode', 'CLIPTextEncodeSDXL') and prompt_patched == 0:
                if 'text' in inputs:
                    inputs['text'] = prompt
                    prompt_patched += 1
                elif 'text_g' in inputs:
                    inputs['text_g'] = prompt
                    inputs['text_l'] = prompt
                    prompt_patched += 1

            if 'KSampler' in ctype or ctype in ('KSamplerAdvanced', 'SamplerCustom'):
                if 'seed' in inputs:
                    inputs['seed'] = seed
                    seeds_patched += 1
                elif 'noise_seed' in inputs:
                    inputs['noise_seed'] = seed
                    seeds_patched += 1

        if prompt_patched == 0:
            raise ComfyError(
                "Video workflow has no CLIPTextEncode node with a 'text' input — "
                "JARVIS doesn't know where to inject the prompt. Re-export the "
                "workflow from ComfyUI with File → Export (API) and make sure "
                "the prompt node is a standard CLIPTextEncode."
            )
        return workflow, {'prompt_nodes': prompt_patched, 'seed_nodes': seeds_patched}

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------
    def generate_image(self, prompt, negative='', checkpoint=None, width=512,
                       height=512, steps=20, cfg=7.0, seed=None,
                       timeout=DEFAULT_IMAGE_TIMEOUT):
        if not prompt or not prompt.strip():
            raise ComfyError("prompt is required")
        seed = seed if seed is not None else int(time.time() * 1000) % (2**31)
        checkpoint = checkpoint or os.environ.get(
            'JARVIS_COMFY_CKPT', 'v1-5-pruned-emaonly.safetensors'
        )

        workflow = self._build_image_workflow(
            prompt.strip(), negative or '', checkpoint,
            max(64, min(int(width), 2048)),
            max(64, min(int(height), 2048)),
            max(1, min(int(steps), 150)),
            float(cfg), int(seed)
        )

        logger.info(f"🎨 ComfyUI image: {prompt[:60]!r} ({width}×{height}, {steps} steps)")
        prompt_id = self._queue(workflow)
        outputs = self._wait(prompt_id, timeout)
        files = self._download_outputs(outputs, f"img_{prompt_id[:8]}")
        return {
            "success": True,
            "prompt_id": prompt_id,
            "seed": seed,
            "checkpoint": checkpoint,
            "files": files
        }

    def generate_video(self, prompt, workflow_path, seed=None,
                       timeout=DEFAULT_VIDEO_TIMEOUT):
        if not prompt or not prompt.strip():
            raise ComfyError("prompt is required")
        if not workflow_path or not os.path.exists(workflow_path):
            raise ComfyError(
                f"Video workflow file not found: {workflow_path}. Export one from "
                f"ComfyUI (File → Export (API)) and save it there."
            )
        try:
            with open(workflow_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
        except Exception as e:
            raise ComfyError(f"Could not parse video workflow JSON: {e}")

        # Exported API workflows are the graph itself. Older exports wrap it
        # in {"prompt": {...}} — unwrap defensively.
        workflow = raw.get('prompt', raw) if isinstance(raw, dict) else raw
        if not isinstance(workflow, dict) or not workflow:
            raise ComfyError("Video workflow JSON is not a non-empty object")

        seed = seed if seed is not None else int(time.time() * 1000) % (2**31)
        workflow, patch_info = self._patch_video_workflow(workflow, prompt.strip(), seed)

        logger.info(f"🎬 ComfyUI video: {prompt[:60]!r} (patched {patch_info})")
        prompt_id = self._queue(workflow)
        outputs = self._wait(prompt_id, timeout)
        files = self._download_outputs(outputs, f"vid_{prompt_id[:8]}")
        return {
            "success": True,
            "prompt_id": prompt_id,
            "seed": seed,
            "patched": patch_info,
            "files": files
        }