import torch
from abc import ABC, abstractmethod


class BaseAttack(ABC):

    def __init__(self):
        self.history = torch.empty((0, 3), dtype=torch.int64)

    @abstractmethod
    def perturb(self, timestamps, src, dst, msg):
        """
        Perform a perturbation attack.

        :param timestamps: timestamps of the transactions
        :param src: source addresses of the transactions
        :param dst: destination addresses of the transactions
        :param msg: message of the transaction

        :return: (p_timestamps, p_src, p_dst): perturbed data
        """
        ...

    def add_entries(self, time, src, dst):
        """
        Add entries to the history.
        """

        entries = torch.stack([time, src, dst], dim=1)
        self.history = torch.cat((self.history, entries))

    def clear_entries(self):
        """
        Clears all entries from the history.
        """

        self.history = torch.empty((0, 3), dtype=torch.int64)

    def get_timestamps(self):
        """
        Return all timestamps from the history.
        """
        return self.history[:, 0]

    def get_src(self):
        """
        Return all source addresses from the history.
        """
        return self.history[:, 1]

    def get_dst(self):
        """
        Return all destination addresses from the history.
        """
        return self.history[:, 2]

    def get_min_timestamp(self):
        """
        Return the minimum timestamp value from the history.
        """
        return self.get_timestamps().min()

    def get_max_timestamp(self):
        """
        Return the maximum timestamp value from the history.
        """
        return self.get_timestamps().max()

    def get_unique_src(self):
        """
        Return all unique source addresses from the history.
        """
        return torch.unique(self.get_src())

    def get_unique_dst(self):
        """
        Return all unique destination addresses from the history.
        """
        return torch.unique(self.get_dst())

    def get_src_count(self):
        """
        Return how many times each source address is present in the history.
        """
        counts = torch.unique(self.get_src(), return_counts=True)
        return torch.stack(counts, dim=1)

    def get_dst_count(self):
        """
        Return how many times each destination address is present in the history.
        """
        counts = torch.unique(self.get_dst(), return_counts=True)
        return torch.stack(counts, dim=1)
