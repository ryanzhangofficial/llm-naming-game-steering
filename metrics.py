import json
import math
import os
import pandas as pd
import numpy as np


def load_logs(path: str) -> pd.DataFrame:
    rows = []
    if os.path.isdir(path):
        for fn in os.listdir(path):
            if fn.endswith('.jsonl'):
                with open(os.path.join(path, fn), 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            x = json.loads(line)
                            x['_file'] = fn
                            rows.append(x)
                        except Exception:
                            continue
    else:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                x = json.loads(line)
                x['_file'] = os.path.basename(path)
                rows.append(x)
    return pd.DataFrame(rows)


def is_agg(df: pd.DataFrame) -> pd.Series:
    return df.get('aggregate', False).fillna(False) == True


def _entropy(p):
    p = np.array(p, dtype=float)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def summarize(df: pd.DataFrame, population: int, target: float = 0.9) -> dict:
    adf = df[is_agg(df)].copy()
    if adf.empty:
        return {"rounds_to_target_mean": np.nan, "rounds_to_target_std": np.nan, "tokens_total_mean": np.nan, "final_entropy_mean": np.nan, "final_entropy_std": np.nan}
    grp = adf.groupby(['seed'])
    rtt = []
    tokens_total = []
    final_entropy = []
    for seed, g in grp:
        g = g.sort_values('round')
        hit = g[g['population_agreement'] >= target]
        if len(hit) == 0:
            rtt.append(np.nan)
        else:
            rtt.append(int(hit['round'].iloc[0]))
        tokens_total.append(float(g['round_tokens'].sum()))
        last_round = int(g['round'].max())
        last_rows = df[(df['seed'] == seed) & (df['round'] == last_round) & (~is_agg(df))]
        names = list(last_rows['i_name'].astype(str)) + list(last_rows['j_name'].astype(str))
        counts = pd.Series(names).value_counts()
        final_entropy.append(_entropy(counts.values))
    return {
        "rounds_to_target_mean": float(np.nanmean(rtt)),
        "rounds_to_target_std": float(np.nanstd(rtt)),
        "tokens_total_mean": float(np.mean(tokens_total)),
        "final_entropy_mean": float(np.mean(final_entropy)),
        "final_entropy_std": float(np.std(final_entropy)),
    }
