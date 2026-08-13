"""Shared Groq client for structured (JSON-schema) outputs.

Ensures GROQ_API_KEY is present, picks a model, and parses responses into
Pydantic models. All functions are synchronous.
"""

from __future__ import annotations

import json
import os
from typing import Type, TypeVar

from dotenv import load_dotenv
from groq import Groq
from groq.types.chat import ChatCompletionMessage
from pydantic import BaseModel

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

DEFAULT_MODEL = "openai/gpt-oss-120b"

T = TypeVar("T", bound=BaseModel)


def _strict_schema(model_cls: Type[BaseModel]) -> dict:
    """Pydantic -> JSON schema with additionalProperties:false on every object.

    Groq's strict json_schema mode rejects schemas that omit this. Field
    defaults are dropped so every property is required, which strict mode
    also demands.
    """
    schema = model_cls.model_json_schema()

    def clean(obj):
        if isinstance(obj, dict):
            if obj.get("type") == "object":
                obj["additionalProperties"] = False
                props = obj.get("properties")
                if props is not None:
                    obj["required"] = sorted(props.keys())
            for value in obj.values():
                clean(value)
        elif isinstance(obj, list):
            for item in obj:
                clean(item)

    clean(schema)
    return schema


def get_client() -> Groq:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return Groq(api_key=key)


def chat_structured(
    model_cls: Type[T],
    system: str,
    user: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
) -> T:
    """Call Groq with JSON-schema structured output and return a parsed model."""
    client = get_client()
    schema = _strict_schema(model_cls)

    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": model_cls.__name__,
                "strict": True,
                "schema": schema,
            },
        },
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    message: ChatCompletionMessage = completion.choices[0].message
    content = message.content or ""
    return model_cls.model_validate_json(content)


def chat_text(system: str, user: str, model: str = DEFAULT_MODEL) -> str:
    """Plain-text Groq call returning the raw assistant text."""
    client = get_client()
    completion = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return completion.choices[0].message.content or ""


def json_loads_robust(text: str) -> dict:
    """Best-effort JSON parse that strips markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return json.loads(text)