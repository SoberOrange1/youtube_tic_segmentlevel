"""
Batch helper for pairwise ranking loss across multiple videos.
"""
from typing import Iterable, Sequence, Tuple, List, Optional
import torch
from .ranking import ranking_loss_for_video

Span = Tuple[int, int]


def ranking_loss_for_batch(
    batch_frame_probs: Sequence[torch.Tensor],
    batch_pos_segments: Sequence[Sequence[Span]],
    batch_neg_segments: Sequence[Sequence[Span]],
    window: int = 5,
    alpha: float = 0.5,
    margin: float = 0.5,
    reduction: str = "mean",
    sample_weight: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """
    Compute average ranking loss over a batch of videos.

    Args:
      batch_frame_probs: list of 1D tensors, len=B.
      batch_pos_segments: list of lists of (start, end).
      batch_neg_segments: list of lists of (start, end).
      window, alpha, margin: hyperparams passed to per-video ranking.
      reduction: 'mean' | 'sum' over samples (inside each sample already reduced).
      sample_weight: optional weights per sample (len=B); used only when reduction='mean'.

    Returns:
      Scalar tensor loss aggregated over the batch.
    """
    assert len(batch_frame_probs) == len(batch_pos_segments) == len(batch_neg_segments), "batch size mismatch"
    B = len(batch_frame_probs)

    losses = []
    for i in range(B):
        l = ranking_loss_for_video(
            batch_frame_probs[i],
            batch_pos_segments[i],
            batch_neg_segments[i],
            window=window,
            alpha=alpha,
            margin=margin,
            reduction="mean",
        )
        losses.append(l)

    if len(losses) == 0:
        # Try to keep device consistent if possible
        return torch.tensor(0.0)

    losses_t = torch.stack(losses)

    if reduction == "sum":
        return losses_t.sum()

    # default mean with optional weights
    if sample_weight is not None:
        w = torch.tensor(sample_weight, dtype=losses_t.dtype, device=losses_t.device)
        w = w / (w.sum() + 1e-8)
        return (losses_t * w).sum()

    return losses_t.mean()
