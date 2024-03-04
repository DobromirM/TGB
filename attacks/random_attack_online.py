import numpy as np

from attacks.base_attack import BaseAttack
from attacks.utils import random_array


class RandomAttackOnline(BaseAttack):

    def __init__(self, attack_dataset="train", rate=0.1, batch_size=200):
        super().__init__(attack_dataset)

        # The rate dictates how many fake samples the attacker should create
        self.rate = rate
        self.batch_size = batch_size

    def __repr__(self):
        return f"RandomAttackOnline(attack_dataset={self.attack_dataset}, rate={self.rate}, batch_size={self.batch_size})"

    def perturb(self, t, src, dst, msg, label):
        # Calculate how many fake entries to generate per batch
        fake_entries_size = int(self.batch_size * self.rate)

        fake_timestamps_all = np.empty([0, ])
        fake_src_all = np.empty([0, ])
        fake_dst_all = np.empty([0, ])
        fake_msg_all = np.empty([0, 1])
        fake_label_all = np.empty([0, ])

        for i in range(self.batch_size, len(t), self.batch_size):
            self.add_entries(t[i - self.batch_size: i], src[i - self.batch_size: i], dst[i - self.batch_size: i])

            unique_src = self.get_unique_src()
            unique_dst = self.get_unique_dst()

            # Generate fake entries for batch
            fake_timestamps = random_array(self.get_min_timestamp(), self.get_max_timestamp(), fake_entries_size)
            fake_src = np.random.choice(unique_src, fake_entries_size, replace=True)
            fake_dst = np.random.choice(unique_dst, fake_entries_size, replace=True)
            fake_msg = msg[np.random.choice(msg.shape[0], fake_entries_size, replace=True)]
            fake_label = np.random.choice(label, fake_entries_size, replace=True)

            # Add them to all fake entries
            fake_timestamps_all = np.concatenate([fake_timestamps_all, fake_timestamps])
            fake_src_all = np.concatenate([fake_src_all, fake_src])
            fake_dst_all = np.concatenate([fake_dst_all, fake_dst])
            fake_msg_all = np.concatenate([fake_msg_all, fake_msg])
            fake_label_all = np.concatenate([fake_label_all, fake_label])

        # Add the fake entries to the original
        t = np.concatenate([t, fake_timestamps_all])
        src = np.concatenate([src, fake_src_all])
        dst = np.concatenate([dst, fake_dst_all])
        msg = np.concatenate([msg, fake_msg_all])
        label = np.concatenate([label, fake_label_all])

        return t, src, dst, msg, label
