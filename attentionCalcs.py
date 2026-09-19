# Manual dot product calculation of two pytorch vectors
import torch


# Encode all vector positions using RoPE
def rope_encode(vec, base=10000.0):

    if vec.ndim != 3:
        raise ValueError(
            "RoPE expects shape [batch, sequence_length, head_dim]"
        )

    batch_size, seq_len, dim = vec.shape

    if dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")

    half_dim = dim // 2

    # Frequencies for each pair of dimensions
    freq_seq = torch.arange(
        half_dim,
        device=vec.device,
        dtype=vec.dtype
    )

    inv_freq = 1.0 / (base ** (freq_seq / half_dim))

    # One position per token in the sequence
    positions = torch.arange(
        seq_len,
        device=vec.device,
        dtype=vec.dtype
    ).unsqueeze(1)

    # Shape: [seq_len, half_dim]
    angles = positions * inv_freq

    cos = torch.cos(angles).unsqueeze(0)
    sin = torch.sin(angles).unsqueeze(0)

    # Pair adjacent dimensions:
    # [x0, x1], [x2, x3], ...
    vec_pairs = vec.reshape(
        batch_size,
        seq_len,
        half_dim,
        2
    )

    even = vec_pairs[..., 0]
    odd = vec_pairs[..., 1]

    rotated = torch.empty_like(vec_pairs)

    rotated[..., 0] = even * cos - odd * sin
    rotated[..., 1] = even * sin + odd * cos

    return rotated.reshape(batch_size, seq_len, dim)


# Encode all vector positions with sinusoidal embeddings
def sinusoidal_encode(vec):
    batch_size, seq_len, dim = vec.shape

    if dim % 2 != 0:
        raise ValueError("Dimension must be even")

    half_dim = dim // 2

    positions = torch.arange(
        seq_len,
        device=vec.device,
        dtype=vec.dtype
    ).unsqueeze(1)

    freq_seq = torch.arange(
        half_dim,
        device=vec.device,
        dtype=vec.dtype
    )

    inv_freq = 1.0 / (10000 ** (freq_seq / half_dim))

    angles = positions * inv_freq

    pe = torch.zeros(
        seq_len,
        dim,
        device=vec.device,
        dtype=vec.dtype
    )

    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)

    return vec + pe.unsqueeze(0)


