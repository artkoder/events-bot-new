import pytest

from main import redact_token


def test_redact_token_long():
    assert redact_token('abcdefghijklmnopqrstuvwxyz') == 'abcdef…wxyz'


def test_redact_token_short():
    assert redact_token('1234567890') == '<redacted>'
