"""LangGraph ReAct agent against the router — the strictest wire check available.

job-hunter's graph does not use LangChain's model layer (DESIGN.md D8/D11), so
this is not about supporting LangGraph. It is here because `create_react_agent`
over `ChatOpenAI(base_url=…)` exercises the exact sequence that was broken:

    turn 1  request carries `tools`      → response must carry `tool_calls`
    turn 2  assistant message with `content: null` + `tool_calls`,
            then a `role: "tool"` message                → must not 422

Run it with its dependencies in a scratch environment — llm-router does not
depend on LangChain and should not start:

    uv run --with langchain-openai --with langgraph python scripts/langgraph_smoke.py

Costs 2-3 provider requests. OpenRouter free is 50/day; do not loop it.
"""
import os
import sys

MODEL = os.getenv("SMOKE_MODEL", "openrouter:nvidia/nemotron-3-nano-30b-a3b:free")
ROUTER = os.getenv("ROUTER_URL", "http://127.0.0.1:8642/v1")


def main() -> int:
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    calls: list[str] = []

    @tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        calls.append(city)
        return f"{city}: 14C, showers"

    agent = create_react_agent(
        ChatOpenAI(
            model=MODEL,
            base_url=ROUTER,
            api_key=os.getenv("ROUTER_KEY", "none"),
            temperature=0,
            timeout=90,
        ),
        [get_weather],
    )

    result = agent.invoke(
        {"messages": [("user", "What is the weather in Melbourne? Use the tool.")]}
    )
    messages = result["messages"]
    final = messages[-1].content

    tool_calls = [m for m in messages if getattr(m, "tool_calls", None)]
    print(f"model      {MODEL}")
    print(f"turns      {len(messages)}")
    print(f"tool_calls {[c['name'] for m in tool_calls for c in m.tool_calls]}")
    print(f"final      {str(final)[:120]}")

    if not calls:
        print("\nFAIL  the tool was never called — `tools` did not reach the model")
        return 1
    if "14" not in str(final):
        print("\nFAIL  the tool result did not make it back into the answer")
        return 1
    print("\nPASS  a full ReAct turn survived the router")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ImportError as e:
        print(f"missing dependency: {e}\nrun with: uv run --with langchain-openai "
              "--with langgraph python scripts/langgraph_smoke.py", file=sys.stderr)
        raise SystemExit(2)
