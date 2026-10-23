"""Prompt templates and the shared message format used by every provider."""
from __future__ import annotations

DEFAULT_TEMPLATES: dict[str, str] = {
    "plain": "{input}",
    "one_sentence": "Answer in one sentence: {input}",
    # Context-specific wording lives in SYSTEM_CONTEXT, so it only applies when context exists.
    "careful": "Answer the question accurately. If you are not certain, say so.\n\nQuestion: {input}",
}

# One instruction for every provider, so the comparison is fair. Only sent when context is given.
SYSTEM_CONTEXT = (
    "Use only the context below to answer. If the context does not contain the answer, "
    "say that the context does not say.\n\nContext:\n{context}"
)


def render(template: str, user_input: str) -> str:
    return template.replace("{input}", user_input.strip())


def build_messages(prompt: str, context: str = "") -> list[dict]:
    messages: list[dict] = []
    if context and context.strip():
        messages.append({"role": "system", "content": SYSTEM_CONTEXT.replace("{context}", context.strip())})
    messages.append({"role": "user", "content": prompt})
    return messages
