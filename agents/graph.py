"""
Academic Multi-Agent System with LangGraph + Qwen

Agents:
  1. Supervisor     → Central router, decides which agent to trigger
  2. Researcher     → Searches papers on ArXiv and Semantic Scholar
  3. Critic         → Analyzes your draft against the literature
  4. Editor         → Reviews and formats in scientific journal style
"""

import os
import sys
from datetime import datetime
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.llm_client import get_qwen_llm
from tools.search_tools import (
    extract_pdf_text,
    fetch_paper_from_url,
    get_arxiv_paper_details,
    get_paper_citations,
    search_arxiv,
    search_semantic_scholar,
)
from tools.memory_store import AcademicMemory

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "config", ".env"))

# Initialize global memory
memory = AcademicMemory()


# ─────────────────────────────────────────────
# STATE DEFINITION
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    """State shared across all agents in the graph."""
    messages: Annotated[list[BaseMessage], add_messages]
    next_agent: str
    draft_text: str        # User's draft text (if provided)
    research_context: str  # Papers found by the Researcher
    iteration: int         # Counter to prevent infinite loops


# ─────────────────────────────────────────────
# AGENT SYSTEM PROMPTS
# ─────────────────────────────────────────────

SUPERVISOR_SYSTEM = """You are the Supervisor of an academic assistance system.
Your role is to analyze the user's request and decide which specialist agent should respond.

Available agents:
- **researcher**: To search for papers, find related literature, explore citations, get article details
- **critic**: To review a user's draft/paper, compare with existing literature, identify gaps
- **editor**: To improve writing, format in IEEE/Nature/ACM style, fix scientific grammar, improve structure
- **direct**: To respond directly without calling another agent (general questions, clarifications)

Reply ONLY with the agent name: researcher, critic, editor, or direct
Nothing else, just the name."""

RESEARCHER_SYSTEM = """You are the Researcher Agent of a specialized academic system.
You have access to the following search tools:
- search_arxiv: Search papers on ArXiv
- search_semantic_scholar: Search on Semantic Scholar with citation metrics
- get_arxiv_paper_details: Full details of an ArXiv paper
- get_paper_citations: Papers that cite an article
- fetch_paper_from_url: Fetch content from a paper URL

When receiving a request:
1. Identify the best search terms (preferably in English)
2. Search multiple sources when relevant
3. Synthesize results highlighting: relevance, year, impact (citations), open-access availability
4. Suggest both highly cited papers AND recent papers
5. When relevant, identify key authors in the field

Be specific, cite paper IDs, and provide useful summaries of findings."""

CRITIC_SYSTEM = """You are the Critic Agent of an academic system. Your role is to critically review
scientific drafts by comparing them with the literature in the field.

When reviewing a draft:
1. **Contribution**: Does the paper have a clear and novel contribution?
2. **Literature**: Does it cite relevant work? Are there obvious bibliographic gaps?
3. **Methodology**: Is the methodology appropriate and well described?
4. **Results**: Are results presented clearly? Are there comparisons with baselines?
5. **Limitations**: Are limitations discussed honestly?
6. **Structure**: Does it follow the standard scientific paper structure?
7. **Language**: Is it academic, precise, and unambiguous?

Use the research context (if available) to compare with the state of the art.
Provide structured, specific, and constructive feedback."""

EDITOR_SYSTEM = """You are the Scientific Editor Agent specialized in formatting and style for top journals.
You know the styles of: IEEE, Nature, Science, ACM, Elsevier, Springer.

When receiving text for editing:
1. Fix grammar and scientific style (especially if in English)
2. Improve terminological precision
3. Strengthen scientific argumentation
4. Format citations correctly (IEEE: [1], APA, etc.)
5. Improve abstracts to be informative and impactful (max. 250 words)
6. Check section structure (Introduction, Related Work, Methodology, Results, Discussion, Conclusion)
7. Suggest improvements for figures/tables when mentioned
8. Adapt the tone to the target venue if specified

Provide improved versions with explanations of changes."""


# ─────────────────────────────────────────────
# AGENT NODE CREATORS
# ─────────────────────────────────────────────

def create_supervisor(llm):
    """Creates the supervisor node that routes requests to specialist agents."""
    def supervisor_node(state: AgentState) -> AgentState:
        messages = state["messages"]
        last_human = next(
            (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
        )
        if not last_human:
            return {**state, "next_agent": "direct"}

        response = llm.invoke([
            SystemMessage(content=SUPERVISOR_SYSTEM),
            HumanMessage(content=f"User request: {last_human.content}")
        ])

        decision = response.content.strip().lower()
        valid = {"researcher", "critic", "editor", "direct"}
        next_agent = decision if decision in valid else "direct"

        return {
            **state,
            "next_agent": next_agent,
            "iteration": state.get("iteration", 0) + 1
        }

    return supervisor_node


def create_researcher_agent(llm):
    """Creates the researcher agent that searches for papers."""
    tools = [
        search_arxiv,
        search_semantic_scholar,
        get_arxiv_paper_details,
        get_paper_citations,
        fetch_paper_from_url
    ]
    agent_llm = llm.bind_tools(tools)

    def researcher_node(state: AgentState) -> AgentState:
        messages = state["messages"]
        last_human = next(
            (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
        )

        # ✨ Retrieve context from memory
        enhanced_system = RESEARCHER_SYSTEM
        if last_human:
            context_from_memory = memory.get_research_context(
                last_human.content,
                k=3
            )

            if context_from_memory and "Previous Research Context" in context_from_memory:
                enhanced_system = (
                    f"{RESEARCHER_SYSTEM}\n\n"
                    f"{context_from_memory}\n\n"
                    f"Use this context to enrich your search and avoid repeating work already done."
                )

        response = agent_llm.invoke([
            SystemMessage(content=enhanced_system),
            *messages
        ])

        # Execute tool calls if any
        tool_results = []
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                tool_map = {t.name: t for t in tools}
                if tc["name"] in tool_map:
                    try:
                        result = tool_map[tc["name"]].invoke(tc["args"])
                        tool_results.append(f"[{tc['name']}]\n{result}")

                        # ✨ Save papers found to vector DB
                        if tc["name"] in ["search_arxiv", "search_semantic_scholar"] and result:
                            memory.add_paper(
                                paper_text=result,
                                metadata={
                                    "source": tc["name"].replace("search_", ""),
                                    "query": tc["args"].get("query", ""),
                                    "timestamp": datetime.now().isoformat(),
                                }
                            )
                    except Exception as e:
                        tool_results.append(f"[{tc['name']}] Error: {str(e)}")

        # Generate final response with tool results
        context = "\n\n".join(tool_results) if tool_results else ""

        final_messages = [
            SystemMessage(content=enhanced_system),
            *messages
        ]
        if context:
            final_messages.append(HumanMessage(
                content=f"Search results:\n\n{context}\n\nNow synthesize and present the findings in a useful way."
            ))

        final_response = llm.invoke(final_messages)

        # ✨ Save conversation to memory
        if last_human:
            memory.add_conversation(
                user_msg=last_human.content,
                agent_response=final_response.content,
                metadata={
                    "agent_type": "researcher",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        research_context = context if context else final_response.content

        return {
            **state,
            "messages": [AIMessage(content=f"🔍 **Researcher Agent:**\n\n{final_response.content}")],
            "research_context": research_context,
        }

    return researcher_node


def create_critic_agent(llm):
    """Creates the critic agent that reviews drafts."""
    tools = [search_arxiv, search_semantic_scholar]
    agent_llm = llm.bind_tools(tools)

    def critic_node(state: AgentState) -> AgentState:
        messages = state["messages"]
        draft = state.get("draft_text", "")
        context = state.get("research_context", "")

        system_msg = CRITIC_SYSTEM
        if draft:
            system_msg += f"\n\n## User Draft to Review:\n{draft}"
        if context:
            system_msg += f"\n\n## Relevant Literature Found:\n{context}"

        # Additional search if needed
        response = agent_llm.invoke([SystemMessage(content=system_msg), *messages])

        tool_results = []
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                tool_map = {t.name: t for t in tools}
                if tc["name"] in tool_map:
                    try:
                        result = tool_map[tc["name"]].invoke(tc["args"])
                        tool_results.append(f"[{tc['name']}]\n{result}")
                    except Exception as e:
                        tool_results.append(f"[{tc['name']}] Error: {str(e)}")

        extra_context = "\n\n".join(tool_results)
        final_system = system_msg
        if extra_context:
            final_system += f"\n\n## Additional Search:\n{extra_context}"

        final = llm.invoke([SystemMessage(content=final_system), *messages])

        # ✨ Save conversation to memory
        last_human = next(
            (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
        )
        if last_human:
            memory.add_conversation(
                user_msg=last_human.content,
                agent_response=final.content,
                metadata={
                    "agent_type": "critic",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        return {
            **state,
            "messages": [AIMessage(content=f"🔬 **Critic Agent:**\n\n{final.content}")],
        }

    return critic_node


def create_editor_agent(llm):
    """Creates the editor agent that formats and improves text."""
    def editor_node(state: AgentState) -> AgentState:
        messages = state["messages"]
        draft = state.get("draft_text", "")

        system_msg = EDITOR_SYSTEM
        if draft:
            system_msg += f"\n\n## Text to Edit:\n{draft}"

        response = llm.invoke([SystemMessage(content=system_msg), *messages])

        # ✨ Save conversation to memory
        last_human = next(
            (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
        )
        if last_human:
            memory.add_conversation(
                user_msg=last_human.content,
                agent_response=response.content,
                metadata={
                    "agent_type": "editor",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        return {
            **state,
            "messages": [AIMessage(content=f"✏️ **Editor Agent:**\n\n{response.content}")],
        }

    return editor_node


def create_direct_response(llm):
    """Creates the direct response agent for general queries."""
    GENERAL_SYSTEM = """You are a specialized academic assistant.
    Respond clearly, precisely, and helpfully. For questions about research,
    scientific methodology, academic writing, or how to use this system,
    provide detailed and practical guidance."""

    def direct_node(state: AgentState) -> AgentState:
        messages = state["messages"]
        response = llm.invoke([SystemMessage(content=GENERAL_SYSTEM), *messages])

        # ✨ Save conversation to memory
        last_human = next(
            (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
        )
        if last_human:
            memory.add_conversation(
                user_msg=last_human.content,
                agent_response=response.content,
                metadata={
                    "agent_type": "direct",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        return {
            **state,
            "messages": [AIMessage(content=response.content)],
        }

    return direct_node


# ─────────────────────────────────────────────
# LANGGRAPH GRAPH BUILDER
# ─────────────────────────────────────────────

def build_academic_graph():
    """Builds and returns the LangGraph graph of the multi-agent system."""
    llm = get_qwen_llm(temperature=0.3)

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("supervisor", create_supervisor(llm))
    graph.add_node("researcher", create_researcher_agent(llm))
    graph.add_node("critic", create_critic_agent(llm))
    graph.add_node("editor", create_editor_agent(llm))
    graph.add_node("direct", create_direct_response(llm))

    # Set entry point
    graph.set_entry_point("supervisor")

    # Conditional routing from supervisor
    def route_from_supervisor(state: AgentState) -> Literal["researcher", "critic", "editor", "direct"]:
        return state["next_agent"]

    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "researcher": "researcher",
            "critic": "critic",
            "editor": "editor",
            "direct": "direct",
        }
    )

    # All agents terminate at END
    graph.add_edge("researcher", END)
    graph.add_edge("critic", END)
    graph.add_edge("editor", END)
    graph.add_edge("direct", END)

    return graph.compile()


# ─────────────────────────────────────────────
# COMMAND-LINE INTERFACE (for testing)
# ─────────────────────────────────────────────

def run_agent(user_message: str, draft_text: str = "", research_context: str = "") -> str:
    """Runs the agent and returns the final response."""
    graph = build_academic_graph()

    initial_state = AgentState(
        messages=[HumanMessage(content=user_message)],
        next_agent="",
        draft_text=draft_text,
        research_context=research_context,
        iteration=0,
    )

    result = graph.invoke(initial_state)

    last_ai = next(
        (m for m in reversed(result["messages"]) if isinstance(m, AIMessage)), None
    )
    return last_ai.content if last_ai else "No response from agent."


if __name__ == "__main__":
    print("🎓 Academic Research Agent — CLI Mode")
    print("=" * 50)
    query = "Search recent papers on malware detection using machine learning"
    print(f"Query: {query}\n")
    response = run_agent(query)
    print(response)