import numpy as np

from attacks.base_attack import BaseAttack
from attacks.utils import random_array
from torch_geometric.utils import degree
from collections import Counter


class ConnectivityAttackSegmented(BaseAttack):

    def __init__(self, attack_dataset="train", rate=0.1, segments = 10):
        super().__init__(attack_dataset)

        # The rate dictates how many fake samples the attacker should create
        self.rate = rate
        self.segments = segments

    def __repr__(self):
        return f"ConnectivityAttackSegmented(attack_dataset={self.attack_dataset}, rate={self.rate})"

    def perturb(self, t, src, dst, msg, label):
        self.add_entries(t, src, dst)

        unique_src = self.get_unique_src()
        unique_dst = self.get_unique_dst()

        # Calculate how many fake entries to generate
        fake_entries_size = int(len(src) * self.rate)

        #calculate how many nodes to choose from 
        src_size = int(len(unique_src) * self.rate)
        dst_size = int(len(unique_dst) * self.rate)

        prev_seg_idx = 0
        # initialize fake entries
        fake_timestamps = []
        fake_src = []
        fake_dst = []
        fake_msg = []
        fake_label = []

        for s in range(self.segments):
            #find the percentile of time stamps according to the timestamp
            p = np.percentile(t, round((100/self.segments)*(s+1)),interpolation='nearest')
            #seg_idx = t.index(p)
            seg_idx = np.where(t == p)[0][0]
            seg_t = t[prev_seg_idx:seg_idx]

            #calculate the degree of each src
            sources_count = Counter(src[prev_seg_idx:seg_idx])
            # Sort the dictionary items by their values in descending order
            sorted_sources = sorted(sources_count.items(), key=lambda x: x[1], reverse=False)
            top_n_src = [item[0] for item in sorted_sources[:src_size]]

            #calculate the degree of each dst
            dst_count = Counter(dst[prev_seg_idx:seg_idx])
            # Sort the dictionary items by their values in descending order
            sorted_dst = sorted(dst_count.items(), key=lambda x: x[1], reverse=False)
            top_n_dist = [item[0] for item in sorted_dst[:dst_size]]

            # Generate fake entries
            fake_timestamps.extend(random_array(prev_seg_idx, seg_idx, round(fake_entries_size/self.segments)))
            fake_src.extend(np.random.choice(top_n_src, round(fake_entries_size/self.segments), replace=True))
            fake_dst.extend(np.random.choice(top_n_dist, round(fake_entries_size/self.segments), replace=True))
            fake_msg.extend(msg[np.random.choice(msg.shape[0], round(fake_entries_size/self.segments), replace=True)])
            fake_label.extend(np.random.choice(label, round(fake_entries_size/self.segments), replace=True))

        # Add the fake entries to the original
        t = np.concatenate([t, fake_timestamps])
        src = np.concatenate([src, fake_src])
        dst = np.concatenate([dst, fake_dst])
        msg = np.concatenate([msg, fake_msg])
        label = np.concatenate([label, fake_label])

        return t, src, dst, msg, label
