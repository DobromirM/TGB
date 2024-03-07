import numpy as np
from tqdm import tqdm

from attacks.base_attack import BaseAttack


class EvolutionaryAttackOnline(BaseAttack):
    def __init__(self, attack_dataset="train", rate=0.1, batch_size=200, mutation_rate=0.25, mutation_range=100):
        super().__init__(attack_dataset)
        self.rate = rate
        self.batch_size = batch_size
        self.mutation_rate = mutation_rate
        self.mutation_range = mutation_range

    def __repr__(self):
        return f"EvolutionaryAttackOnline(attack_dataset={self.attack_dataset}, rate={self.rate}, batch_size={self.batch_size}, mutation_rate={self.mutation_rate}, mutation_range={self.mutation_range})"

    def perturb(self, t, src, dst, msg, label):
        fake_entries_size = int(self.batch_size * self.rate)

        fake_timestamps_all = np.empty([0, ])
        fake_src_all = np.empty([0, ])
        fake_dst_all = np.empty([0, ])
        fake_msg_all = np.empty([0, 1])
        fake_label_all = np.empty([0, ])

        for i in tqdm(range(self.batch_size, len(t), self.batch_size)):
            self.add_entries(t[i - self.batch_size: i], src[i - self.batch_size: i], dst[i - self.batch_size: i])

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
            for src_i, dst_i in zip(src_idxs, dst_idxs):
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
                    fake_timestamp = fake_timestamp + np.random.randint(low=-self.mutation_range,
                                                                        high=self.mutation_range,
                                                                        size=1)

                fake_timestamps = np.concatenate([fake_timestamps, fake_timestamp])

            fake_msg = msg[np.random.choice(msg.shape[0], fake_entries_size, replace=True)]
            fake_label = np.random.choice(label, fake_entries_size, replace=True)

            fake_timestamps_all = np.concatenate([fake_timestamps_all, fake_timestamps.flatten()])
            fake_src_all = np.concatenate([fake_src_all, fake_src])
            fake_dst_all = np.concatenate([fake_dst_all, fake_dst])
            fake_msg_all = np.concatenate([fake_msg_all, fake_msg])
            fake_label_all = np.concatenate([fake_label_all, fake_label])

        t = np.concatenate([t, fake_timestamps_all])
        src = np.concatenate([src, fake_src_all])
        dst = np.concatenate([dst, fake_dst_all])
        msg = np.concatenate([msg, fake_msg_all])
        label = np.concatenate([label, fake_label_all])

        return t, src, dst, msg, label
