"""ComfyUI client. Images come back over the websocket (SaveImageWebsocket node), so
nothing is written to disk by ComfyUI; the caller decides what to keep after screening."""
import json
import time
import uuid
from urllib.parse import urlparse

import requests
import websocket

from . import safety

SAVE_NODE = "9"


class ComfyError(RuntimeError):
    pass


def _options(spec) -> list[str]:
    first = spec[0]
    if first == "COMBO" and isinstance(spec[1], dict):
        return spec[1].get("options", [])
    return first if isinstance(first, list) else []


class Comfy:
    def __init__(self, cfg):
        self.c = cfg.comfy
        self.sizes = cfg.comfy["sizes"]
        self.url = self.c["url"].rstrip("/")
        self.ws_url = f"ws://{urlparse(self.url).netloc}/ws"
        self.client_id = uuid.uuid4().hex
        self.lora_names: set[str] = set()
        self.version: str | None = None

    def check(self):
        try:
            info = requests.get(f"{self.url}/object_info", timeout=60).json()
        except requests.RequestException as e:
            raise ComfyError(f"ComfyUI is not reachable at {self.url} ({e}). Start ComfyUI first.") from e
        if "SaveImageWebsocket" not in info:
            raise ComfyError("ComfyUI is missing the SaveImageWebsocket node (custom_nodes/websocket_image_save.py). "
                             "It is required so images stay in memory until they pass safety screening.")
        ckpts = _options(info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"])
        if self.c["checkpoint"] not in ckpts:
            raise ComfyError(f"checkpoint {self.c['checkpoint']!r} not found in ComfyUI. Available: {ckpts}")
        self.lora_names = set(_options(info["LoraLoader"]["input"]["required"]["lora_name"]))
        try:
            self.version = requests.get(f"{self.url}/system_stats", timeout=30).json()["system"]["comfyui_version"]
        except (requests.RequestException, KeyError, ValueError):
            self.version = None

    def size(self, aspect: str) -> tuple[int, int]:
        """Named aspect from config ("square"/"portrait"/"landscape") or an explicit "WxH"."""
        if "x" in aspect:
            w, h = aspect.split("x")
            return int(w), int(h)
        w, h = self.sizes.get(aspect, self.sizes["square"])
        return int(w), int(h)

    def settings(self) -> dict:
        """Everything besides prompt/seed/weight/size needed to reproduce a render."""
        return {"checkpoint": self.c["checkpoint"], "steps": self.c["steps"], "cfg": self.c["cfg"],
                "sampler": self.c["sampler"], "scheduler": self.c["scheduler"], "clip_skip": self.c["clip_skip"],
                "denoise": 1.0, "comfyui_version": self.version}

    def workflow(self, positive: str, negative: str, seed: int, aspect: str,
                 lora_name: str | None, weight: float, *, mode: str) -> dict:
        # Safety hardening is applied here so no caller can bypass it; the mode is required.
        positive, negative = safety.harden(positive, negative, mode)
        w, h = self.size(aspect)
        model, clip = ["1", 0], ["1", 1]
        wf = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": self.c["checkpoint"]}}}
        if lora_name:
            wf["2"] = {"class_type": "LoraLoader", "inputs": {
                "model": model, "clip": clip, "lora_name": lora_name,
                "strength_model": weight, "strength_clip": weight}}
            model, clip = ["2", 0], ["2", 1]
        wf.update({
            "3": {"class_type": "CLIPSetLastLayer", "inputs": {"clip": clip, "stop_at_clip_layer": self.c["clip_skip"]}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": positive}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": negative}},
            "6": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
            "7": {"class_type": "KSampler", "inputs": {
                "model": model, "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0],
                "seed": seed, "steps": self.c["steps"], "cfg": self.c["cfg"],
                "sampler_name": self.c["sampler"], "scheduler": self.c["scheduler"], "denoise": 1.0}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}},
            SAVE_NODE: {"class_type": "SaveImageWebsocket", "inputs": {"images": ["8", 0]}},
        })
        return wf

    def render(self, workflow: dict) -> bytes:
        """Queue one workflow and return the PNG bytes of its single output image."""
        ws = websocket.WebSocket()
        ws.connect(f"{self.ws_url}?clientId={self.client_id}", timeout=30)
        try:
            r = requests.post(f"{self.url}/prompt", json={"prompt": workflow, "client_id": self.client_id}, timeout=60)
            if r.status_code != 200:
                raise ComfyError(f"ComfyUI rejected the workflow: {r.text[:800]}")
            prompt_id = r.json()["prompt_id"]
            deadline = time.time() + self.c["timeout_s"]
            current_node, images = None, []
            while True:
                if time.time() > deadline:
                    raise ComfyError(f"timed out waiting for prompt {prompt_id}")
                try:
                    msg = ws.recv()
                except websocket.WebSocketTimeoutException:
                    continue
                if isinstance(msg, str):
                    m = json.loads(msg)
                    data = m.get("data") or {}
                    if data.get("prompt_id") != prompt_id:
                        continue
                    if m["type"] == "executing":
                        current_node = data.get("node")
                        if current_node is None:
                            break
                    elif m["type"] == "execution_error":
                        raise ComfyError(f"ComfyUI error: {data.get('exception_message', data)}")
                    elif m["type"] == "execution_success":
                        break
                elif current_node == SAVE_NODE and len(msg) > 8:
                    images.append(bytes(msg[8:]))  # 4-byte event type + 4-byte image format header
            if not images:
                raise ComfyError(f"prompt {prompt_id} finished without an image")
            return images[-1]
        finally:
            ws.close()
