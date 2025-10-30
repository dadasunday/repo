from src.main import generate_voice


def test_generate_voice_returns_bytes():
    out = generate_voice("hello")
    assert isinstance(out, (bytes, bytearray))
    assert out == b"hello"


def test_generate_voice_raises_on_none():
    import pytest

    with pytest.raises(ValueError):
        generate_voice(None)
