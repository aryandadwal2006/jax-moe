import numpy as np

VOCAB = tuple("0123456789+=") + ("_",)
TOKEN_TO_ID = {token: index for index, token in enumerate(VOCAB)}
PAD_ID = TOKEN_TO_ID["_"]


def make_dataset(size: int, max_digits: int = 3, seed: int = 0) -> np.ndarray:
    if size < 1 or max_digits < 1:
        raise ValueError("size and max_digits must be positive")
    rng = np.random.default_rng(seed)
    sequence_length = 3 * max_digits + 3
    data = np.full((size, sequence_length), PAD_ID, dtype=np.int32)
    for row in range(size):
        digits = int(rng.integers(1, max_digits + 1))
        lower = 0 if digits == 1 else 10 ** (digits - 1)
        upper = 10 ** digits
        left = int(rng.integers(lower, upper))
        right = int(rng.integers(lower, upper))
        sequence = f"{left}+{right}={left + right}"
        encoded = [TOKEN_TO_ID[token] for token in sequence]
        data[row, :len(encoded)] = encoded
    return data
