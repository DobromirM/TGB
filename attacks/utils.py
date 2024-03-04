import numpy as np


def random_array(min_val, max_val, n):
    """
    Generate an array containing `n` random values between `min_val` and `max_val`
    """
    return np.random.randint(low=min_val, high=max_val, size=(n,))
