"""
LLM client for Qwen via OpenRouter or DashScope.
Compatible with the OpenAI interface for use with LangChain.
"""
import os
from langchain_openai import ChatOpenAI


def get_qwen_llm(temperature: float = 0.3) -> ChatOpenAI:
    """
    Returns a LangChain-compatible Qwen client.
    Supports OpenRouter (recommended) or DashScope directly.
    """
    # Try OpenRouter first
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    dashscope_key = os.getenv("QWEN_API_KEY", "")

    if openrouter_key and openrouter_key != "your_openrouter_key_here":
        return ChatOpenAI(
            model=os.getenv("QWEN_MODEL", "qwen/qwen3-235b-a22b"),
            api_key=openrouter_key,
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            temperature=temperature,
            default_headers={
                "HTTP-Referer": "https://academic-agent.local",
                "X-Title": "Academic Research Agent"
            }
        )
    elif dashscope_key:
        return ChatOpenAI(
            model=os.getenv("QWEN_MODEL", "qwen-plus"),
            api_key=dashscope_key,
            base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            temperature=temperature,
        )
    else:
        raise EnvironmentError(
            "No API key found!\n"
            "Set OPENROUTER_API_KEY or QWEN_API_KEY in the .env file.\n"
            "See config/.env for instructions."
        )
