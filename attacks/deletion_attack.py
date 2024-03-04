import torch
import random

from attacks.utils import random_tensor

from attacks.base_attack import BaseAttack



class RandomDelAttack(BaseAttack):

    def __init__(self, rate=0.1):
        super().__init__()

        # The rate dictates how many samples the attacker should delete
        self.rate = rate

    def perturb(self, t, src, dst, msg):
        self.clear_entries()
        self.add_entries(t, src, dst)

        unique_src = self.get_unique_src()
        unique_dst = self.get_unique_dst()

        # Calculate how many entries to delete
        fake_entries_size = int(len(src) * self.rate)

        del_idx = random.sample(range(0, len(src)), fake_entries_size)
        t = t[t!=del_idx]
        src = src[src!=del_idx]
        dst = dst[dst!=del_idx]
        msg = msg[msg!=del_idx]

        return t, src, dst, msg
