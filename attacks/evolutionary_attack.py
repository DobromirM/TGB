import numpy as np
from tqdm import tqdm

from attacks.base_attack import BaseAttack


class EvolutionaryAttack(BaseAttack):
    def __init__(self, attack_dataset="train", rate=0.1, mutation_rate=0.25, mutation_range=100):
        super().__init__(attack_dataset)
        self.rate = rate
        self.mutation_rate = mutation_rate
        self.mutation_range = mutation_range

    def __repr__(self):
        return f"EvolutionaryAttack(attack_dataset={self.attack_dataset}, rate={self.rate}, mutation_rate={self.mutation_rate}, mutation_range={self.mutation_range})"

    def perturb(self, t, src, dst, msg, label):
        self.add_entries(t, src, dst)

        fake_entries_size = int(len(src) * self.rate)

        # Calculate the score of each sample
        src_degree = np.bincount(self.get_src().astype(int))[self.get_src().astype(int)]
        dst_degree = np.bincount(self.get_dst().astype(int))[self.get_dst().astype(int)]
        scores = src_degree + dst_degree

        selected_idxs = scores.argsort()[:fake_entries_size]
        src_idxs = np.random.choice(selected_idxs, fake_entries_size)
        dst_idxs = np.random.choice(selected_idxs, fake_entries_size)

        fake_timestamps = np.empty([0, ])
        fake_src = self.get_src()[src_idxs]
        fake_dst = self.get_dst()[dst_idxs]

        # crossover
        for src_i, dst_i in tqdm(zip(src_idxs, dst_idxs), total=len(dst_idxs)):
            if src[src_i] == dst[dst_i]:
                continue

            if t[src_i] < t[dst_i]:
                fake_timestamp = np.random.randint(low=t[src_i], high=t[dst_i], size=1)
            elif t[src_i] > t[dst_i]:
                fake_timestamp = np.random.randint(low=t[dst_i], high=t[src_i], size=1)
            else:
                fake_timestamp = t[src_i]

            # mutation
            if np.random.randint(0, 1) < self.mutation_rate:
                fake_timestamp = fake_timestamp + np.random.randint(low=-self.mutation_range, high=self.mutation_range,
                                                                    size=1)

            fake_timestamps = np.concatenate([fake_timestamps, fake_timestamp])

        fake_msg = msg[np.random.choice(msg.shape[0], fake_entries_size, replace=True)]
        fake_label = np.random.choice(label, fake_entries_size, replace=True)

        t = np.concatenate([t, fake_timestamps])
        src = np.concatenate([src, fake_src])
        dst = np.concatenate([dst, fake_dst])
        msg = np.concatenate([msg, fake_msg])
        label = np.concatenate([label, fake_label])

        return t, src, dst, msg, label