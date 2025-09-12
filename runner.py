import argparse
import os
import time
from datetime import datetime
from utils import ensure_dir
from env import run_game
from llm import LLMWrapper, MockLLM


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-path')
    p.add_argument('--condition', choices=['nl', 'nl_sw', 'schema'])
    p.add_argument('--population-size', type=int, default=24)
    p.add_argument('--rounds', type=int, default=100)
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--n-lexicon', type=int, default=12)
    p.add_argument('--memory-k', type=int, default=5)
    p.add_argument('--payload-limit', type=int, default=20)
    p.add_argument('--max-new-tokens', type=int, default=32)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top-p', type=float, default=0.9)
    p.add_argument('--repeat-penalty', type=float, default=1.1)
    p.add_argument('--base-seed', type=int, default=42)
    p.add_argument('--lose-shift-alpha', type=float, default=0.75)
    p.add_argument('--quiet', action='store_true')
    p.add_argument('--mock', action='store_true')
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--wandb-project', type=str, default='sign-naming-game')
    p.add_argument('--wandb-offline', action='store_true')
    p.add_argument('--ablation', action='store_true')
    p.add_argument('--ablation-populations', type=str, help='Comma-separated list, e.g., 12,24')
    p.add_argument('--ablation-memory', type=str, help='Comma-separated list, e.g., 5,10')
    p.add_argument('--ablation-alpha', type=str, help='Comma-separated list, e.g., 0.5,0.75,0.9')
    args = p.parse_args()
    if not args.ablation and not args.condition:
        raise SystemExit('Missing --condition (required unless --ablation is provided)')
    ensure_dir('data')
    ensure_dir('figs')
    if args.mock:
        llm = MockLLM()
    else:
        if not args.model_path:
            raise SystemExit('Missing --model-path')
        llm = LLMWrapper(model_path=args.model_path)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    if args.ablation:
        # Defaults for ablation grids
        pop_list = [int(x) for x in (args.ablation_populations.split(',') if args.ablation_populations else ['12','24'])]
        mem_list = [int(x) for x in (args.ablation_memory.split(',') if args.ablation_memory else ['5','10'])]
        alpha_list = [float(x) for x in (args.ablation_alpha.split(',') if args.ablation_alpha else ['0.5','0.75','0.9'])]
        conditions = ['nl_sw', 'schema']
        rounds = 300
        seeds = 3
        total_runs = 0
        # Count runs for reporting
        total_runs += len(pop_list) * len(alpha_list)  # nl (no memory)
        total_runs += len(pop_list) * len(mem_list) * len(alpha_list)  # nl_sw
        total_runs += len(pop_list) * len(mem_list) * len(alpha_list)  # schema
        print(f"Ablation mode: total runs={total_runs}")
        for cond in conditions:
            for N in pop_list:
                if cond == 'nl':
                    for alpha in alpha_list:
                        cfg = {
                            'population': N,
                            'rounds': rounds,
                            'seeds': seeds,
                            'condition': cond,
                            'n_lexicon': args.n_lexicon,
                            'memory_k': args.memory_k,  # unused in 'nl'
                            'payload_limit': args.payload_limit,
                            'max_new_tokens': args.max_new_tokens,
                            'temperature': args.temperature,
                            'top_p': args.top_p,
                            'repeat_penalty': args.repeat_penalty,
                            'base_seed': args.base_seed,
                            'lose_shift_alpha': alpha,
                            'quiet': args.quiet,
                        }
                        out_name = f"logs_ablate_{cond}_N{N}_R{rounds}_S{seeds}_alpha{alpha}_{ts}.jsonl"
                        out_path = os.path.join('data', out_name)
                        run = None
                        if args.wandb:
                            try:
                                if args.wandb_offline:
                                    os.environ['WANDB_MODE'] = 'offline'
                                import wandb
                                run_name = f"abl_{cond}_N{N}_R{rounds}_S{seeds}_alpha{alpha}_{ts}"
                                run = wandb.init(project=args.wandb_project, name=run_name, config=cfg, reinit=True)
                            except Exception:
                                run = None
                        t0 = time.time()
                        path = run_game(cfg, llm, out_path, wandb_run=run)
                        dt = time.time() - t0
                        print(path)
                        print(f"elapsed_sec={dt:.2f}")
                        if run is not None:
                            try:
                                run.finish()
                            except Exception:
                                pass
                else:
                    for K in mem_list:
                        for alpha in alpha_list:
                            cfg = {
                                'population': N,
                                'rounds': rounds,
                                'seeds': seeds,
                                'condition': cond,
                                'n_lexicon': args.n_lexicon,
                                'memory_k': K,
                                'payload_limit': args.payload_limit,
                                'max_new_tokens': args.max_new_tokens,
                                'temperature': args.temperature,
                                'top_p': args.top_p,
                                'repeat_penalty': args.repeat_penalty,
                                'base_seed': args.base_seed,
                                'lose_shift_alpha': alpha,
                                'quiet': args.quiet,
                            }
                            out_name = f"logs_ablate_{cond}_N{N}_R{rounds}_S{seeds}_K{K}_alpha{alpha}_{ts}.jsonl"
                            out_path = os.path.join('data', out_name)
                            run = None
                            if args.wandb:
                                try:
                                    if args.wandb_offline:
                                        os.environ['WANDB_MODE'] = 'offline'
                                    import wandb
                                    run_name = f"abl_{cond}_N{N}_R{rounds}_S{seeds}_K{K}_alpha{alpha}_{ts}"
                                    run = wandb.init(project=args.wandb_project, name=run_name, config=cfg, reinit=True)
                                except Exception:
                                    run = None
                            t0 = time.time()
                            path = run_game(cfg, llm, out_path, wandb_run=run)
                            dt = time.time() - t0
                            print(path)
                            print(f"elapsed_sec={dt:.2f}")
                            if run is not None:
                                try:
                                    run.finish()
                                except Exception:
                                    pass
        return

    # Single-run mode
    cfg = {
        'population': args.population_size,
        'rounds': args.rounds,
        'seeds': args.seeds,
        'condition': args.condition,
        'n_lexicon': args.n_lexicon,
        'memory_k': args.memory_k,
        'payload_limit': args.payload_limit,
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'repeat_penalty': args.repeat_penalty,
        'base_seed': args.base_seed,
        'lose_shift_alpha': args.lose_shift_alpha,
        'quiet': args.quiet,
    }
    out_name = f"logs_{args.condition}_N{args.population_size}_R{args.rounds}_S{args.seeds}_{ts}.jsonl"
    out_path = os.path.join('data', out_name)
    run = None
    if args.wandb:
        try:
            if args.wandb_offline:
                os.environ['WANDB_MODE'] = 'offline'
            import wandb
            run_name = f"{args.condition}_N{args.population_size}_R{args.rounds}_S{args.seeds}_{ts}"
            run = wandb.init(project=args.wandb_project, name=run_name, config=cfg, reinit=True)
        except Exception:
            run = None
    t0 = time.time()
    path = run_game(cfg, llm, out_path, wandb_run=run)
    dt = time.time() - t0
    print(path)
    print(f"elapsed_sec={dt:.2f}")
    if run is not None:
        try:
            run.finish()
        except Exception:
            pass


if __name__ == '__main__':
    main()
