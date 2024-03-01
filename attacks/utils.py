import torch


def random_tensor(min_val, max_val, n):
    """
    Generate a tensor containing `n` random values between `min_val` and `max_val`
    """
    return torch.randint(low=min_val, high=max_val, size=(n,), dtype=torch.int64)
