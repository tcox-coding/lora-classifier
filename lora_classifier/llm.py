import base64
import io
import json
import re
import time

import requests
from PIL import Image


class LLMError(RuntimeError):
    pass


class RefusalError(LLMError):
    pass


_REFUSAL_RE = re.compile(r"\b(I can(no|')t|I am unable|I'm unable|I won't|cannot assist|can't help with)\b", re.I)


def parse_json(text: str):
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


class LLM:
    def __init__(self, cfg):
        if not cfg.api_key:
            raise LLMError("No DeepInfra API key: set DEEPINFRA_API_KEY or add it to .env")
        self.d = cfg.deepinfra
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {cfg.api_key}"})
        self.total_cost = 0.0
        # Independent vision model that re-checks images the screener flags ("" disables it).
        self.second_opinion_model = self.d.get("second_opinion_model") or None

    def image_part(self, png_bytes: bytes) -> dict:
        im = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        im.thumbnail((self.d["image_max_side"], self.d["image_max_side"]))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
        return {"type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}}

    def chat_json(self, system: str, content: list | str, max_tokens: int | None = None, retries: int = 3,
                  model: str | None = None):
        body = {
            "model": model or self.d["model"],
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "temperature": self.d["temperature"],
            "max_tokens": max_tokens or self.d["max_tokens"],
            "response_format": {"type": "json_object"},
        }
        last_err = None
        for attempt in range(retries):
            try:
                r = self.session.post(f"{self.d['base_url']}/chat/completions", json=body, timeout=self.d["timeout_s"])
                if r.status_code in (429, 500, 502, 503, 504):
                    raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
                if r.status_code != 200:
                    raise LLMError(f"HTTP {r.status_code}: {r.text[:800]}")  # not retryable
                data = r.json()
                self.total_cost += float((data.get("usage") or {}).get("estimated_cost") or 0)
                text = data["choices"][0]["message"].get("content") or ""
                try:
                    return parse_json(text)
                except json.JSONDecodeError:
                    if _REFUSAL_RE.search(text[:300]):
                        raise RefusalError(text[:300])
                    raise LLMError(f"model returned non-JSON: {text[:300]}")
            except RefusalError:
                raise
            except (requests.RequestException, LLMError) as e:
                last_err = e
                if isinstance(e, LLMError) and str(e).startswith("HTTP 4") and "HTTP 429" not in str(e):
                    raise
                time.sleep(3 * (attempt + 1) ** 2)
        raise LLMError(f"DeepInfra call failed after {retries} attempts: {last_err}")
