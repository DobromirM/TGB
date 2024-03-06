import numpy as np
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

from attacks import BaseAttack


class ConstrainedAttack(BaseAttack):
    def __init__(self, attack_dataset="train", rate=0.1, kde_bandwidth=0.1, time_window=100,
                 max_node_degree_strat='median'):
        super().__init__(attack_dataset)
        self.rate = rate
        self.kde_bandwidth = kde_bandwidth
        self.time_window = time_window
        self.max_node_degree_strat_name = max_node_degree_strat

        if max_node_degree_strat == "mean":
            self.max_node_strat = np.mean
        elif max_node_degree_strat == "median":
            self.max_node_strat = np.median
        else:
            raise Exception("Invalid max node strategy!")

    def __repr__(self):
        return f"ConstrainedAttack(attack_dataset={self.attack_dataset}, rate={self.rate}, kde_bandwidth={self.kde_bandwidth}, time_window={self.time_window}, max_node_degree_strat={self.max_node_degree_strat_name})"

    def perturb(self, t, src, dst, msg, label):
        self.add_entries(t, src, dst)

        src_count = self.get_src_count()
        dst_count = self.get_dst_count()

        # Calculate how many fake entries to generate
        fake_entries_size = int(len(src) * self.rate)

        # Create the timestamps distribution
        time_dist = (KernelDensity(bandwidth=self.kde_bandwidth, kernel="gaussian"))
        time_dist = time_dist.fit(self.get_timestamps().reshape(-1, 1))

        # Sample the fake timestamps from the distribution
        fake_timestamps = np.round(time_dist.sample(fake_entries_size))

        fake_src_all = np.empty([0, ])
        fake_dst_all = np.empty([0, ])

        unique_fake_timestamps = np.unique(fake_timestamps, return_counts=True)

        # Generate fake data for each unique timestamp based on the time and node requirements
        for timestamp, count in tqdm(zip(*unique_fake_timestamps), total=len(unique_fake_timestamps[0])):
            time_filter = np.logical_and(t <= timestamp, t >= timestamp - self.time_window)

            # src
            src_candidates = np.unique(self.get_src()[time_filter])
            src_candidate_counts = src_count[np.isin(src_count[:, 0], src_candidates)]
            max_src_nodes_degree = self.max_node_strat(src_candidate_counts[:, 1])
            src_candidate_counts = src_candidate_counts[src_candidate_counts[:, 1] <= max_src_nodes_degree]
            src_candidates = src_candidate_counts[:, 0]
            fake_src = np.random.choice(src_candidates, count, replace=True)
            fake_src_all = np.concatenate([fake_src_all, fake_src])

            # dst
            dst_candidates = np.unique(self.get_dst()[time_filter])
            dst_candidate_counts = dst_count[np.isin(dst_count[:, 0], dst_candidates)]
            max_dst_nodes_degree = self.max_node_strat(dst_candidate_counts[:, 1])
            dst_candidate_counts = dst_candidate_counts[dst_candidate_counts[:, 1] <= max_dst_nodes_degree]
            dst_candidates = dst_candidate_counts[:, 0]
            fake_dst = np.random.choice(dst_candidates, count, replace=True)
            fake_dst_all = np.concatenate([fake_dst_all, fake_dst])

        # Generate fake msgs and labels randomly
        fake_msg = msg[np.random.choice(msg.shape[0], fake_entries_size, replace=True)]
        fake_label = np.random.choice(label, fake_entries_size, replace=True)

        t = np.concatenate([t, fake_timestamps.flatten()])
        src = np.concatenate([src, fake_src_all])
        dst = np.concatenate([dst, fake_dst_all])
        msg = np.concatenate([msg, fake_msg])
        label = np.concatenate([label, fake_label])

        return t, src, dst, msg, label
