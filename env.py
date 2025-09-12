import json
import time
import math
import random
from collections import Counter, defaultdict
from typing import Dict, List
from utils import set_seeds, ensure_dir, Memory
from agents import Agent


def _pair_indices(n: int, rng: random.Random) -> List[tuple]:
    idx = list(range(n))
    rng.shuffle(idx)
    pairs = []
    for i in range(0, n, 2):
        if i + 1 < n:
            pairs.append((idx[i], idx[i + 1]))
    return pairs


def run_game(cfg: Dict, llm, out_path: str, wandb_run=None) -> str:
    ensure_dir("data")
    set_seeds(int(cfg["base_seed"]))
    N = int(cfg["population"])
    R = int(cfg["rounds"])
    seeds = int(cfg["seeds"])
    condition = str(cfg["condition"]).lower()
    n_lex = int(cfg["n_lexicon"])
    memory_k = int(cfg["memory_k"])
    payload_limit = int(cfg["payload_limit"])
    max_new_tokens = int(cfg["max_new_tokens"])
    temperature = float(cfg["temperature"])
    top_p = float(cfg["top_p"])
    repeat_penalty = float(cfg["repeat_penalty"])
    base_seed = int(cfg["base_seed"])
    lose_shift_alpha = float(cfg.get("lose_shift_alpha", 0.75))
    quiet = bool(cfg.get("quiet", False))

    lexicon = [f"C{i+1}" for i in range(n_lex)]

    t0 = time.time()
    path = out_path
    f = open(path, "w", encoding="utf-8")
    try:
        for s in range(seeds):
            seed_int = int(base_seed) ^ (s * 2654435761) ^ (N * 97531) ^ (R * 131071)
            rng = random.Random(seed_int & 0xFFFFFFFF)
            agents = [Agent(i, llm, condition, lexicon, payload_limit, seed=base_seed + s) for i in range(N)]
            memories = [Memory(memory_k) for _ in range(N)]
            # Initialize preferred names z_i uniformly from lexicon
            z = [rng.choice(lexicon) for _ in range(N)]
            for r in range(R):
                pairs = _pair_indices(N, rng)
                names_counter = Counter()
                round_tokens = 0
                pair_success = 0
                for i, j in pairs:
                    # Agents propose their current preferred names z_i and z_j
                    name_i, raw_i, tok_i, comp_i = agents[i].propose(r, z[i], max_new_tokens, temperature, top_p, repeat_penalty, base_seed + s * 100000)
                    name_j, raw_j, tok_j, comp_j = agents[j].propose(r, z[j], max_new_tokens, temperature, top_p, repeat_penalty, base_seed + s * 100000 + 1)
                    # Count only valid names for population agreement
                    if name_i is not None:
                        names_counter.update([name_i])
                    if name_j is not None:
                        names_counter.update([name_j])
                    round_tokens += tok_i + tok_j
                    success = (name_i is not None) and (name_j is not None) and (name_i == name_j)
                    if success:
                        pair_success += 1
                    rec = {
                        "seed": s,
                        "round": r,
                        "pair": [i, j],
                        "i_id": i,
                        "j_id": j,
                        "i_name": name_i,
                        "j_name": name_j,
                        "i_txt": raw_i,
                        "j_txt": raw_j,
                        "i_tokens": tok_i,
                        "j_tokens": tok_j,
                        "i_compliant": comp_i if condition == "schema" else None,
                        "j_compliant": comp_j if condition == "schema" else None,
                        "condition": condition,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    # Update rules per condition
                    # Partner-only memory updates apply to 'nl_sw' and 'schema'; plain 'nl' skips modal update
                    def modal_from_memory(mem: Memory) -> str:
                        items = [x for x in mem.list() if isinstance(x, str)]
                        if not items:
                            return None
                        counts = Counter(items)
                        maxc = max(counts.values())
                        cands = [k for k, v in counts.items() if v == maxc]
                        def idx(x: str) -> int:
                            try:
                                return int(x[1:])
                            except Exception:
                                return 10**9
                        cands.sort(key=lambda x: idx(x))
                        return cands[0] if cands else None

                    if condition in ("nl", "nl_sw", "schema"):
                        # Partner-only memory: each agent stores the partner's decoded name if decodable
                        if name_j is not None:
                            memories[i].add(name_j)
                        if name_i is not None:
                            memories[j].add(name_i)
                        next_i = modal_from_memory(memories[i]) or z[i]
                        next_j = modal_from_memory(memories[j]) or z[j]
                    else:
                        # 'nl' condition: no memory-based modal update
                        next_i = z[i]
                        next_j = z[j]

                    # Additionally, apply win-stay / lose-shift
                    alpha = lose_shift_alpha
                    if success:
                        next_i = z[i]
                        next_j = z[j]
                    else:
                        if name_j is not None and rng.random() < alpha:
                            next_i = name_j
                        if name_i is not None and rng.random() < alpha:
                            next_j = name_i
                    z[i] = next_i
                    z[j] = next_j
                if names_counter:
                    modal = names_counter.most_common(1)[0][1]
                else:
                    modal = 0
                pop_agree = modal / N
                agg = {
                    "seed": s,
                    "round": r,
                    "aggregate": True,
                    "pairs": len(pairs),
                    "round_tokens": round_tokens,
                    "pair_success": pair_success,
                    "population_agreement": pop_agree,
                    "condition": condition,
                }
                f.write(json.dumps(agg, ensure_ascii=False) + "\n")
                if wandb_run is not None:
                    try:
                        step = s * R + r
                        wandb_run.log({
                            "seed": s,
                            "round": r,
                            "pairs": len(pairs),
                            "round_tokens": round_tokens,
                            "pair_success": pair_success,
                            "population_agreement": pop_agree,
                            "condition": condition,
                        }, step=step)
                    except Exception:
                        pass
                if not quiet and (r + 1) % max(1, R // 10) == 0:
                    pass
    finally:
        f.close()
    return path
