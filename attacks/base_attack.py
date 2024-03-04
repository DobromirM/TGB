import numpy as np
import torch
from abc import ABC, abstractmethod


class BaseAttack(ABC):

    def __init__(self, full_data, train_mask, val_mask, test_mask):
        self.full_data = full_data

        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

        self.train_data = dict()
        self.val_data = dict()
        self.test_data = dict()

        for key, value in full_data.items():
            self.train_data[key] = value[train_mask]
            self.val_data[key] = value[val_mask]
            self.test_data[key] = value[test_mask]

        self.history = torch.empty((0, 3), dtype=torch.int64)

    def update_data(self):
        for key, _ in self.full_data.items():
            self.full_data[key] = np.concatenate([self.train_data[key], self.val_data[key], self.test_data[key]])

    def update_masks(self):
        train_len = len(self.train_data['timestamps'])
        val_len = len(self.val_data['timestamps'])
        test_len = len(self.test_data['timestamps'])

        self.train_mask = np.concatenate(
            [np.full(train_len, True), np.full(val_len, False), np.full(test_len, False)])

        self.val_mask = np.concatenate(
            [np.full(train_len, False), np.full(val_len, True), np.full(test_len, False)])

        self.test_mask = np.concatenate(
            [np.full(train_len, False), np.full(val_len, False), np.full(test_len, True)])

    @abstractmethod
    def perturb(self, timestamps, src, dst, msg, label):
        """
        Perform a perturbation attack.

        :param timestamps: timestamps of the transactions
        :param src: source addresses of the transactions
        :param dst: destination addresses of the transactions
        :param msg: message of the transaction
        :param label: label of the transaction

        :return: (p_timestamps, p_src, p_dst, p_msg): perturbed data
        """
        ...

    def perturb_train(self):
        """
        Perform a perturbation attack on the training dataset.
        """

        p_timestamps, p_src, p_dst, p_msg, p_label = self.perturb(self.train_data["timestamps"],
                                                                  self.train_data["sources"],
                                                                  self.train_data["destinations"],
                                                                  self.train_data["edge_feat"],
                                                                  self.train_data["edge_label"])

        sorted_indices = p_timestamps.argsort()

        self.train_data["timestamps"] = p_timestamps[sorted_indices]
        self.train_data["sources"] = p_src[sorted_indices]
        self.train_data["destinations"] = p_dst[sorted_indices]
        self.train_data["edge_feat"] = p_msg[sorted_indices]
        self.train_data["edge_label"] = p_label[sorted_indices]

        self.update_data()
        self.update_masks()

    def perturb_validation(self):
        """
        Perform a perturbation attack on the validation dataset.
        """

        pass

    def perturb_test(self):
        """
        Perform a perturbation attack on the test dataset.
        """

        pass

    def get_masks(self):
        """
        Return the masks for the test, validation and train datasets.
        :return: (train_mask, val_mask, test_mask)
        """

        return self.train_mask, self.val_mask, self.test_mask

    def add_entries(self, time, src, dst):
        """
        Add entries to the history.
        """

        entries = np.stack([time, src, dst], axis=1)
        self.history = np.concatenate((self.history, entries))

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
        return np.unique(self.get_src())

    def get_unique_dst(self):
        """
        Return all unique destination addresses from the history.
        """
        return np.unique(self.get_dst())

    def get_src_count(self):
        """
        Return how many times each source address is present in the history.
        """
        counts = np.unique(self.get_src(), return_counts=True)
        return np.stack(counts, axis=1)

    def get_dst_count(self):
        """
        Return how many times each destination address is present in the history.
        """
        counts = np.unique(self.get_dst(), return_counts=True)
        return np.stack(counts, axis=1)
