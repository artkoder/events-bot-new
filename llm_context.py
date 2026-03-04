"""Runtime context helpers for routing LLM incidents to the right operator chat.

We want operational LLM incidents (rate-limit, provider errors, missing RPC routes)
to be visible to the operator who initiated the action via Telegram UI.

Implementation uses contextvars so it works across async call stacks without
plumbing chat_id through every function.
"""

from __future__ import annotations

import contextvars
from typing import Optional


_operator_chat_id_var: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "operator_chat_id", default=None
)


def get_operator_chat_id() -> Optional[int]:
    return _operator_chat_id_var.get()


def set_operator_chat_id(chat_id: int) -> contextvars.Token:
    return _operator_chat_id_var.set(int(chat_id))


def reset_operator_chat_id(token: contextvars.Token) -> None:
    _operator_chat_id_var.reset(token)

