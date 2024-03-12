import numpy as np

from attacks.base_attack import BaseAttack
from attacks.utils import random_array
import random


class RandomAttackDeletion(BaseAttack):

    def __init__(self, attack_dataset="train", rate=0.1):
        super().__init__(attack_dataset)

        # The rate dictates how many fake samples the attacker should create
        self.rate = rate

    def __repr__(self):
        return f"RandomAttack(attack_dataset={self.attack_dataset}, rate={self.rate})"

    def perturb(self, t, src, dst, msg, label):
        self.clear_entries()
        self.add_entries(t, src, dst)

        # Calculate how many entries to delete
        fake_entries_size = int(len(src) * self.rate)

        keep_idx = random.sample(range(0, len(src)), len(src) - fake_entries_size)
        t = t[keep_idx]
        src = src[keep_idx]
        dst = dst[keep_idx]
        msg = msg[keep_idx]
        label = label[keep_idx]

        return t, src, dst, msg,label