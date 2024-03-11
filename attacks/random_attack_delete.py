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

        del_idx = random.sample(range(0, len(src)), fake_entries_size)
        t = t[t!=del_idx].reshape(-1)
        src = src[src!=del_idx].reshape(-1)
        dst = dst[dst!=del_idx].reshape(-1)
        msg = msg[msg!=del_idx]
        msg = msg.reshape(msg.shape[1], msg.shape[2])
        label = label[label!=del_idx].reshape(-1)

        return t, src, dst, msg,label