"""M2.5 agents. Each is a small LangGraph subgraph with its own role and prompt, built around an
injected chat model, so it can be run and tested on its own with a fake model."""

import json
import re


def parse_json(text: str, default):
    """The first JSON object or list in a model reply, or `default` if there isn't one.

    Qwen sometimes wraps JSON in prose or a ```json fence, so search for it instead of
    expecting the reply to be pure JSON.
    """
    for pattern in (r"\[.*\]", r"\{.*\}"):
        if isinstance(default, dict) and pattern.startswith(r"\["):
            continue
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return default
