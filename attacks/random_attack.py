import torch

from attacks.utils import random_tensor

from attacks.base_attack import BaseAttack


class RandomAttack(BaseAttack):

    def __init__(self, rate=0.1, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()

        # The rate dictates how many fake samples the attacker should create
        self.rate = rate

    def perturb(self, t, src, dst, msg):
        #self.clear_entries()
        #self.add_entries(t, src, dst)

        unique_src = self.get_unique_src()
        unique_dst = self.get_unique_dst()

        # Calculate how many fake entries to generate
        fake_entries_size = int(len(src) * self.rate)

        # We use the same weight for all indices
        fake_src_weights = torch.ones(len(unique_src), )
        fake_dst_weights = torch.ones(len(unique_dst), )

        # Generate fake entries
        # `torch.multinomial` samples random indices using the provided weights
        fake_timestamps = random_tensor(self.get_min_timestamp(), self.get_max_timestamp(), fake_entries_size)
        fake_src = unique_src[torch.multinomial(fake_src_weights, fake_entries_size, replacement=True)]
        fake_dst = unique_dst[torch.multinomial(fake_dst_weights, fake_entries_size, replacement=True)]
        fake_msg = msg[:fake_entries_size]  # Currently not used and contains the same value.

        # Add the fake entries to the original
        t = torch.cat([t, fake_timestamps])
        src = torch.cat([src, fake_src])
        dst = torch.cat([dst, fake_dst])
        msg = torch.cat([msg, fake_msg])

        return t, src, dst, msg
