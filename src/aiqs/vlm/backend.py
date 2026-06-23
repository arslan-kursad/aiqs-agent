import json
import re
import time
import base64
from io import BytesIO
from typing import Literal
from PIL import Image
from pydantic import BaseModel, Field, ValidationError
from aiqs.vlm.state import VLMState

Verdict = Literal["defect", "clean", "unsure"]

class VLMVerdict(BaseModel):
    model_config = {"extra": "forbid"}
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str

def parse_verdict(raw: str) -> VLMVerdict:
    text = raw.strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    data = json.loads(match.group(0)) if match else json.loads(text)
    return VLMVerdict(**data)

class MockVLMBackend:
    def __call__(self, state: VLMState) -> VLMVerdict:
        return VLMVerdict(verdict="clean", confidence=0.9, reasoning="Mock result")

class AnthropicVLMBackend:
    def __init__(self, model="claude-3-5-sonnet-latest"):
        self.model = model
        self._client = None

    def _client_lazy(self):
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic(max_retries=5)
        return self._client

    def __call__(self, state: VLMState) -> VLMVerdict:
        client = self._client_lazy()
        img = Image.open(state.image_path)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        msg = client.messages.create(
            model=self.model,
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_str}},
                    {"type": "text", "text": "Classify this image. Output JSON: {\"verdict\": \"clean|defect|unsure\", \"confidence\": float, \"reasoning\": string}"}
                ]
            }]
        )
        return parse_verdict(msg.content[0].text)
