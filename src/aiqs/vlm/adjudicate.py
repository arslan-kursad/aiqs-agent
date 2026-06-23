from aiqs.vlm.state import VLMState
from aiqs.vlm.backend import AnthropicVLMBackend, VLMVerdict

def adjudicate(state: VLMState) -> VLMVerdict:
    """Default VLM adjudication for an item."""
    # We use the Anthropic backend directly as the mock check was broken
    return AnthropicVLMBackend()(state)
