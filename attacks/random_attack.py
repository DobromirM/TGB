import numpy as np

from attacks.base_attack import BaseAttack
from attacks.utils import random_array


class RandomAttack(BaseAttack):

    def __init__(self, attack_dataset="train", rate=0.1):
        super().__init__(attack_dataset)

        # The rate dictates how many fake samples the attacker should create
        self.rate = rate

    def __repr__(self):
        return f"RandomAttack(attack_dataset={self.attack_dataset}, rate={self.rate})"

    def perturb(self, t, src, dst, msg, label):
        self.add_entries(t, src, dst)

        unique_src = self.get_unique_src()
        unique_dst = self.get_unique_dst()

        # Calculate how many fake entries to generate
        fake_entries_size = int(len(src) * self.rate)

        # Generate fake entries
        fake_timestamps = random_array(self.get_min_timestamp(), self.get_max_timestamp(), fake_entries_size)
        fake_src = np.random.choice(unique_src, fake_entries_size, replace=True)
        fake_dst = np.random.choice(unique_dst, fake_entries_size, replace=True)
        fake_msg = msg[np.random.choice(msg.shape[0], fake_entries_size, replace=True)]
        fake_label = np.random.choice(label, fake_entries_size, replace=True)

        # Add the fake entries to the original
        t = np.concatenate([t, fake_timestamps])
        src = np.concatenate([src, fake_src])
        dst = np.concatenate([dst, fake_dst])
        msg = np.concatenate([msg, fake_msg])
        label = np.concatenate([label, fake_label])

        return t, src, dst, msg, label
