def mem_render(entries):
    return " | ".join(entries[-5:]) if entries else ""


def nl_prompt(agent_id: int, round_id: int, proposed_name: str) -> str:
    t = []
    t.append(f"You are agent {agent_id} in round {round_id}.")
    t.append(f"Your current proposed name is {proposed_name}.")
    t.append("Reply EXACTLY one line proposing your name.")
    return "\n".join(t)


def schema_prompt(agent_id: int, round_id: int, proposed_name: str, payload_limit: int) -> str:
    t = []
    t.append(f"You are agent {agent_id} in round {round_id}.")
    t.append(f"Your current proposed name is {proposed_name}.")
    t.append(f"Reply EXACTLY one line as: @say {{name: {proposed_name}}} | rationale.")
    return "\n".join(t)
