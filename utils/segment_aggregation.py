"""
Utilities for aggregating per-frame tic probabilities into a segment-level score.
"""
from typing import Iterable, List, Sequence, Tuple
import torch


def _ensure_1d(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 0:
        return t.view(1)
    if t.dim() > 1:
        return t.view(-1)
    return t


def score_segment(
    probs: torch.Tensor,
    window: int = 5,
    alpha: float = 0.5,
    clamp: bool = True,
) -> torch.Tensor:
    """
    Compute a segment-level score from per-frame tic probabilities.

    Strategy:
      - Find the peak frame probability within the segment.
      - Compute local average in a symmetric window around that peak.
      - Return a convex combination: score = alpha * peak + (1 - alpha) * local_avg.

    Args:
      probs: 1D tensor of per-frame probabilities for a segment (values in [0, 1]).
      window: neighborhood size (number of frames) for local average (>=1). If even, it is made odd.
      alpha: blend factor between peak and local avg (0..1). alpha=1 uses only max; 0 uses only local avg.
      clamp: if True, clamp probs to [0, 1].

    Returns:
      A scalar tensor (segment score).
    """
    x = _ensure_1d(probs)
    if clamp:
        x = x.clamp(0.0, 1.0)
    n = x.numel()
    if n == 0:
        return torch.tensor(0.0, dtype=x.dtype, device=x.device)

    # Ensure odd window and >=1
    w = max(int(window), 1)
    if w % 2 == 0:
        w += 1
    half = w // 2

    peak_val, peak_idx = torch.max(x, dim=0)

    start = int(max(0, peak_idx.item() - half))
    end = int(min(n, peak_idx.item() + half + 1))

    local_avg = x[start:end].mean() if end > start else peak_val

    if alpha <= 0:
        return local_avg
    if alpha >= 1:
        return peak_val
    return alpha * peak_val + (1.0 - alpha) * local_avg


def score_segment_from_frames(
    frame_probs: torch.Tensor,
    seg_span: Tuple[int, int],
    window: int = 5,
    alpha: float = 0.5,
    clamp: bool = True,
) -> torch.Tensor:
    """
    Convenience wrapper to score a segment given full-sequence frame probabilities.

    Args:
      frame_probs: 1D tensor of probabilities for the whole video/clip.
      seg_span: (start, end) indices [start, end) for the segment within frame_probs.
      window, alpha, clamp: see score_segment.

    Returns:
      Scalar tensor score for the segment.
    """
    s, e = seg_span
    s = max(0, int(s))
    e = max(s, int(e))
    return score_segment(frame_probs[s:e], window=window, alpha=alpha, clamp=clamp)


def batched_score_segments(
    frame_probs: torch.Tensor,
    segments: Sequence[Tuple[int, int]],
    window: int = 5,
    alpha: float = 0.5,
    clamp: bool = True,
) -> torch.Tensor:
    """
    Score multiple segments from a common per-frame probability sequence.

    Args:
      frame_probs: 1D tensor length T.
      segments: sequence of (start, end) spans [start, end).

    Returns:
      Tensor of shape (len(segments),) with segment scores.
    """
    scores = [
        score_segment_from_frames(frame_probs, span, window=window, alpha=alpha, clamp=clamp)
        for span in segments
    ]
    if len(scores) == 0:
        return torch.empty(0, dtype=frame_probs.dtype, device=frame_probs.device)
    return torch.stack(scores, dim=0)
