from typing import Optional
from neuralcore.agents.agent_core import AgentRunner

async def run_agent_loop(
    client,
    prompt,
    dynamic_manager,
    system_prompt,
    context_manager,
    max_iterations: int = 25,
    max_tokens: int = 12000,
) -> bool:
    runner = AgentRunner(
        client,
        max_iterations=max_iterations,
        default_max_tokens=max_tokens,
    )

    success = False

    async for event_type, payload in runner.run(
        user_prompt=prompt,
        messages_so_far=[],
        tools=dynamic_manager,
        system_prompt=system_prompt,
        context_manager=context_manager,
        max_tokens=max_tokens,
    ):
        if event_type == "content_delta":
            print(payload, end="", flush=True)

        elif event_type == "tool_start":
            print(f"\n🔧 {payload['name']} {payload['args']}")

        elif event_type == "tool_result":
            res = payload.get("result", "")
            if payload.get("error"):
                print(f"\n❌ {res}")
            else:
                res_str = str(res)
                print(f"\n✅ {res_str[:300]}{'...' if len(res_str) > 300 else ''}")

        elif event_type == "final_answer":
            success = True
            print("\n\n" + "=" * 80)
            print("🤖 FINAL ANSWER")
            print("=" * 80)
            print(payload)
            print("=" * 80)

        elif event_type in ("error", "warning", "cancelled"):
            print(f"\n[{event_type.upper()}] {payload}")

        elif event_type == "needs_confirmation":
            print("\n⚠️  Needs confirmation - skipping in headless mode")

    # 👇 Clear shell-friendly status line
    print("\n" + "=" * 80)
    print(f"STATUS: {'SUCCESS' if success else 'FAILED'}")
    print("=" * 80)

    return success