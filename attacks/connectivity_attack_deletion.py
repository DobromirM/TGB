import numpy as np

from attacks.base_attack import BaseAttack
from attacks.utils import random_array
import random
from collections import Counter


class ConnectivityAttackDeletion(BaseAttack):

    def __init__(self, attack_dataset="train", rate=0.1):
        super().__init__(attack_dataset)

        # The rate dictates how many fake samples the attacker should create
        self.rate = rate

    def __repr__(self):
        return f"ConnectivityAttackDeletion(attack_dataset={self.attack_dataset}, rate={self.rate})"

    def perturb(self, t, src, dst, msg, label):
        self.clear_entries()
        self.add_entries(t, src, dst)

        unique_src = self.get_unique_src()
        unique_dst = self.get_unique_dst()

        # Calculate how many entries to delete
        fake_entries_size = int(len(src) * self.rate)

        #calculate how many nodes to choose from 
        src_size = int(len(unique_src) * self.rate*3)
        dst_size = int(len(unique_dst) * self.rate)

        #calculate the degree of each src
        sources_count = Counter(src)
        # Sort the dictionary items by their values in descending order
        sorted_sources = sorted(sources_count.items(), key=lambda x: x[1], reverse=False)
        top_n_src = [item[0] for item in sorted_sources[:src_size]]

        # Find indices of elements from src that are also in top_n_src
        common_indices = [i for i, x in enumerate(src) if x in top_n_src]

        # Randomly select 10 distinct indices from common_indices
        del_idx = random.sample(common_indices, min(fake_entries_size, len(common_indices)))
        all_idx = list(range(len(t)))
        keep_idx = [x for x in all_idx if x not in  del_idx]

        t = t[keep_idx]
        src = src[keep_idx]
        dst = dst[keep_idx]
        msg = msg[keep_idx]
        label = label[keep_idx]

        return t, src, dst, msg,label