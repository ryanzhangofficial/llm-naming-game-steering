import os
import random
import string
import numpy as np


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def gen_nonce_codes(n: int, seed: int) -> list:
    rng = np.random.RandomState(seed)
    consonants = list("bcdfghjklmnpqrstvwxyz")
    vowels = list("aeiou")
    out = set()
    while len(out) < n:
        c1 = rng.choice(consonants)
        v = rng.choice(vowels)
        c2 = rng.choice(consonants)
        out.add((c1 + v + c2).upper())
    return list(out)


class Memory:
    def __init__(self, k: int):
        self.k = int(k)
        self.buf = []

    def add(self, item):
        self.buf.append(item)
        if len(self.buf) > self.k:
            self.buf = self.buf[-self.k :]

    def list(self) -> list:
        return list(self.buf)
