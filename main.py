#!/usr/bin/env python3
"""
Academic Research Agent — Entry Point
======================================
Multi-Agent System for Academic Research
LangGraph + Qwen + ArXiv + Semantic Scholar

Usage:
    python main.py           # Starts the web interface (Gradio)
    python main.py --cli     # Command-line mode
    python main.py --test    # Tests the API connection
"""

import os
import sys
import argparse
from dotenv import load_dotenv

# Load .env from the config/ directory
load_dotenv(os.path.join(os.path.dirname(__file__), "config", ".env"))


def test_connection():
    """Tests whether the API is working."""
    print("🔌 Testing connection to the Qwen API...\n")
    try:
        from config.llm_client import get_qwen_llm
        llm = get_qwen_llm()
        response = llm.invoke("Diga apenas: 'Conexão OK!'")
        print(f"✅ Connection successful!\nResponse: {response.content}")
    except EnvironmentError as e:
        print(f"❌ Configuration error:\n{e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Connection error:\n{e}")
        sys.exit(1)


def run_cli():
    """Interactive CLI mode."""
    print("🎓 Academic Research Agent — CLI")
    print("=" * 50)
    print("Commands: 'exit' to quit, 'clear' to reset context\n")

    from agents.graph import build_academic_graph
    from langchain_core.messages import HumanMessage, AIMessage

    graph = build_academic_graph()
    research_context = ""

    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye! 👋")
                break
            if user_input.lower() == "clear":
                research_context = ""
                print("✅ Context cleared.")
                continue

            print("\n⏳ Processing...\n")

            result = graph.invoke({
                "messages": [HumanMessage(content=user_input)],
                "next_agent": "",
                "draft_text": "",
                "research_context": research_context,
                "iteration": 0,
            })

            if result.get("research_context"):
                research_context = result["research_context"][-3000:]

            last_ai = next(
                (m for m in reversed(result["messages"]) if isinstance(m, AIMessage)), None
            )

            if last_ai:
                print(f"🤖 Agent: {last_ai.content}")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye! 👋")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def run_web():
    from ui.app import build_interface, CUSTOM_CSS
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Academic Research Agent")
    parser.add_argument("--cli", action="store_true", help="Command-line mode")
    parser.add_argument("--test", action="store_true", help="Test API connection")
    args = parser.parse_args()

    if args.test:
        test_connection()
    elif args.cli:
        run_cli()
    else:
        run_web()
