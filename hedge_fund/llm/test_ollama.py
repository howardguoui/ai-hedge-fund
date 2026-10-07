"""Local models through Ollama: routing, no key, OpenAI-compatible endpoint, <think> stripping."""

from __future__ import annotations

from hedge_fund.llm import extract_json, make_llm
from hedge_fund.llm.registry import env_var_for, is_supported, provider_for


def test_ollama_prefix_routes_to_ollama_without_a_key():
    assert provider_for("ollama/qwen3:8b") == "Ollama"
    assert provider_for("ollama/llama3.1:8b") == "Ollama"  # any local model, no registry edit
    assert is_supported("Ollama") and env_var_for("Ollama") is None


def test_make_llm_points_at_the_local_server(monkeypatch):
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/")
    llm = make_llm("ollama/qwen3:8b", timeout=30)
    chat = llm._chat if hasattr(llm, "_chat") else llm.chat
    assert chat.model_name == "qwen3:8b"
    assert str(chat.openai_api_base).rstrip("/") == "http://127.0.0.1:11434/v1"
    assert chat.request_timeout >= 600  # local models get a long timeout


def test_think_blocks_are_ignored_when_parsing():
    text = '<think>maybe {"signal": "bearish"}? no.</think>\n{"signal": "bullish", "confidence": 70, "reasoning": "ok"}'
    assert extract_json(text)["signal"] == "bullish"
