"""
Pairwise hinge ranking loss for tic vs non-tic segment scoring.
"""
from typing import Sequence, Tuple, List
import torch
from utils.segment_aggregation import batched_score_segments

Span = Tuple[int, int]


def pairwise_ranking_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    margin: float = 0.5,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute L_rank = sum_{i,j} max(0, m - (s_i - s_j)) over all pos-neg pairs.

    Args:
      pos_scores: tensor of shape (P,)
      neg_scores: tensor of shape (N,)
      margin: hinge margin m
      reduction: 'mean' | 'sum' | 'none'
    """
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return torch.zeros((), dtype=pos_scores.dtype, device=pos_scores.device)

    pos = pos_scores.view(-1, 1)  # (P, 1)
    neg = neg_scores.view(1, -1)  # (1, N)
    # Broadcast to (P, N)
    loss = (margin - (pos - neg)).clamp_min(0.0)

    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    # default mean; guard empty
    return loss.mean() if loss.numel() > 0 else torch.zeros((), dtype=pos_scores.dtype, device=pos_scores.device)


def ranking_loss_for_video(
    frame_probs: torch.Tensor,
    pos_segments: Sequence[Span],
    neg_segments: Sequence[Span],
    window: int = 5,
    alpha: float = 0.5,
    margin: float = 0.5,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Convenience wrapper: compute pos/neg segment scores from per-frame probabilities
    then apply pairwise hinge ranking loss.

    Args:
      frame_probs: 1D tensor of per-frame probabilities in [0,1].
      pos_segments: list of (start, end) for tic segments.
      neg_segments: list of (start, end) for non-tic segments.
      window, alpha: aggregation hyperparams (see utils.segment_aggregation.score_segment).
      margin, reduction: loss hyperparams.
    """
    device = frame_probs.device
    dtype = frame_probs.dtype

    pos_scores = batched_score_segments(frame_probs, pos_segments, window=window, alpha=alpha)
    neg_scores = batched_score_segments(frame_probs, neg_segments, window=window, alpha=alpha)

    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return torch.zeros((), dtype=dtype, device=device)

    return pairwise_ranking_loss(pos_scores, neg_scores, margin=margin, reduction=reduction)


essential_doc = None  # kept to avoid linter warnings if module imported without direct use
