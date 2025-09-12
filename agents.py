import random
from typing import Tuple
from prompts import nl_prompt, schema_prompt
from schema_enforce import parse_schema, extract_nl_name


class Agent:
    def __init__(self, agent_id: int, llm, condition: str, lexicon: list, payload_limit: int, seed: int):
        self.agent_id = agent_id
        self.llm = llm
        self.condition = condition
        self.lexicon = list(lexicon)
        self.payload_limit = payload_limit
        self.rng = random.Random(seed ^ (agent_id * 1315423911))

    def _fallback(self) -> str:
        return self.rng.choice(self.lexicon)

    def propose(self, round_id: int, proposed_name: str, max_new_tokens: int, temperature: float, top_p: float, repeat_penalty: float, base_seed: int) -> Tuple[str, str, int, bool]:
        prompt = None
        raw = None
        name = None
        compliant = None
        if self.condition == "schema":
            prompt = schema_prompt(self.agent_id, round_id, proposed_name, self.payload_limit)
            valid = set(x.upper() for x in self.lexicon)
            raw = self.llm.generate(prompt, max_new_tokens, temperature, top_p, repeat_penalty, seed=base_seed ^ (self.agent_id * 1013904223) ^ (round_id * 1664525))
            n1, ok1 = parse_schema(raw)
            if n1 and n1.upper() in valid:
                name = n1.upper()
                compliant = ok1
            else:
                # Single reminder retry
                remind = "Follow EXACTLY one line: @say {name: Ck}"
                raw2 = self.llm.generate(prompt + "\n" + remind, max_new_tokens, temperature, top_p, repeat_penalty, seed=(base_seed + 997) ^ (self.agent_id * 1013904223) ^ (round_id * 1664525))
                n2, ok2 = parse_schema(raw2)
                if n2 and n2.upper() in valid:
                    raw = raw2
                    name = n2.upper()
                    compliant = ok2
                else:
                    # Try to salvage a valid name from free text if present; otherwise undecodable (None)
                    salv = extract_nl_name(raw2, valid) or extract_nl_name(raw, valid)
                    name = salv if salv else None
                    compliant = False
        else:
            prompt = nl_prompt(self.agent_id, round_id, proposed_name)
            raw = self.llm.generate(prompt, max_new_tokens, temperature, top_p, repeat_penalty, seed=base_seed ^ (self.agent_id * 1013904223) ^ (round_id * 1664525))
            valid = set([x.upper() for x in self.lexicon])
            picked = extract_nl_name(raw, valid)
            name = picked if picked else None
            compliant = True
        tokens = self.llm.tokenize_count(raw)
        return name, raw, tokens, compliant
