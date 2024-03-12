import numpy as np

from attacks.base_attack import BaseAttack
from attacks.utils import random_array
from torch_geometric.utils import degree
from collections import Counter


class ConnectivityAttack(BaseAttack):

    def __init__(self, attack_dataset="train", rate=0.1):
        super().__init__(attack_dataset)

        # The rate dictates how many fake samples the attacker should create
        self.rate = rate

    def __repr__(self):
        return f"ConnectivityAttack(attack_dataset={self.attack_dataset}, rate={self.rate})"

    def perturb(self, t, src, dst, msg, label):
        self.add_entries(t, src, dst)

        unique_src = self.get_unique_src()
        unique_dst = self.get_unique_dst()

        # Calculate how many fake entries to generate
        fake_entries_size = int(len(src) * self.rate)

        #calculate how many nodes to choose from 
        src_size = int(len(unique_src) * self.rate)
        dst_size = int(len(unique_dst) * self.rate)

        #calculate the degree of each src
        sources_count = Counter(src)
        # Sort the dictionary items by their values in descending order
        sorted_sources = sorted(sources_count.items(), key=lambda x: x[1], reverse=False)
        top_n_src = [item[0] for item in sorted_sources[:src_size]]

        #calculate the degree of each dst
        dst_count = Counter(dst)
        # Sort the dictionary items by their values in descending order
        sorted_dst = sorted(dst_count.items(), key=lambda x: x[1], reverse=False)
        top_n_dist = [item[0] for item in sorted_dst[:dst_size]]

        # Generate fake entries
        fake_timestamps = random_array(self.get_min_timestamp(), self.get_max_timestamp(), fake_entries_size)
        fake_src = np.random.choice(top_n_src, fake_entries_size, replace=True)
        fake_dst = np.random.choice(top_n_dist, fake_entries_size, replace=True)
        fake_msg = msg[np.random.choice(msg.shape[0], fake_entries_size, replace=True)]
        fake_label = np.random.choice(label, fake_entries_size, replace=True)

        # Add the fake entries to the original
        t = np.concatenate([t, fake_timestamps])
        src = np.concatenate([src, fake_src])
        dst = np.concatenate([dst, fake_dst])
        msg = np.concatenate([msg, fake_msg])
        label = np.concatenate([label, fake_label])

        return t, src, dst, msg, label
