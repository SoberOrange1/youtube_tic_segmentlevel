import os
import torch
import torch.nn.functional as F
import pickle
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from models import LateFusionModel, ViTFrameEncoder, FrameProbAvgFusionModel
from datasets import TicNonTicSegmentDataset
from visualization import plot_confusion_matrix, plot_roc_curve
import logging
from tqdm import tqdm
# added imports for ranking & aggregation
from losses.ranking import pairwise_ranking_loss
from utils.segment_aggregation import score_segment
from collections import defaultdict
import random
from torch.utils.data import Sampler

# Set random seeds for reproducibility
def set_random_seeds(seed):
    """
    Set random seeds for all random number generators to ensure reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Keys to retain in minimal inference config (Strategy 1)
INFER_CONFIG_KEYS = [
    'vit_model_name','vit_use_lora','vit_lora_r','vit_lora_alpha','vit_lora_dropout','vit_lora_targets',
    'fusion_mode','vit_tcn_config','blend_tcn_config','blend_dim','num_classes',
    'agg_threshold','rank_window','rank_alpha'
]

# Configuration
config = {
    'exp': 'exp_5_CE',
    'segment_root': r'/hdd/shujun/youtube_finetune/data_folder/cropped_faces',
    'results_dir': r'/hdd/shujun/youtube_finetune/results',
    'epochs': 150,
    'batch_size': 8,
    'lr': 1e-4,
    'num_workers': 0,
    'target_len': 30,
    'checkpoint_interval': 5,  # save checkpoint every N epochs (customizable)
    
    # Set random seed for reproducibility
    'seed': 42,

    # threshold sensitivity analysis
    'threshold_analysis': True,  
    'threshold_start': 0.1,      
    'threshold_end': 0.9,        
    'threshold_step': 0.05,
    'threshold_fine_search': True,  # Enable fine-grained search around best threshold
    'threshold_fine_multiplier': 0.2,  # Fine search step = threshold_step * multiplier (n < 1)

    # dataset/twostream options
    'use_blend': True,
    'blend_dim': 52,
    'strict_blend': False,
    'blend_folder_candidates': ['blendshapes', 'landmarks'],
    'blend_exts': ['.npy', '.txt', '.csv'],

    'num_classes': 2,
    'class_names': ['non-tic', 'tic'],

    'vit_model_name': 'trpakov/vit-face-expression',
    'vit_dropout': 0.4,
    # LoRA fine-tuning options (enable query/value adaptation only)
    'vit_use_lora': True,            # set True to activate LoRA
    'vit_lora_r': 8,
    'vit_lora_alpha': 32,
    'vit_lora_dropout': 0.1,
    'vit_lora_targets': ['query', 'value'],  # only adapt attention Q & V
    'vit_tcn_config': {
        'num_channels': [256, 512],
        'kernel_size': 3,
        'dropout': 0.3,
        'causal': True,
        'use_norm': 'weight_norm',
        'activation': 'relu',
        'output_projection': 256,
        'input_shape': 'NCL',
        'pooling_type': 'none'
    },
    'blend_tcn_config': {
        'num_inputs': 52,
        'num_channels': [64, 128, 128],
        'kernel_size': 3,
        'dropout': 0.1,
        'causal': True,
        'use_norm': 'batch_norm',
        'activation': 'gelu',
        'output_projection': 128,
        'input_shape': 'NCL',
        'pooling_type': 'none'
    },
    'fusion_mode': 'concat',
    'start_fold': 1,

    # ranking / aggregation hyperparameters
    'rank_use': False,
    'rank_window': 5,   # aggregation window for score_segment (segment-level)
    'rank_alpha': 0.1,  # peak vs local average blend for score_segment, the value's smaller, the more local avg contributes
    'rank_margin': 0.5, # hinge margin between positive/negative segment scores
    # Weight for temporal smoothness loss (L_smooth). L_total = CE + L_rank + smooth_lambda * L_smooth
    'smooth_lambda':0.0,
    # Weight for cross-entropy loss in total loss. Set to 0.0 to disable CE contribution.
    'ce_lambda': 1,
    'rank_target_neg_per_pos': 4,  # desired number of negatives per positive (per video); will repeat if fewer
    'rank_cache_max_neg_per_video': 64,  # NEW: max cached negatives per video id
    'agg_threshold': 0.5,  # threshold on aggregated tic score for classification

    'model_variant': 'frame_prob_avg',  
    'rank_log_pair_videos': True,  # NEW: print matched / unmatched positive video IDs each epoch
    'early_stop_patience': 10,  # NEW: stop if no val acc improvement after N epochs per fold
    'debug_loss': False,
} 

def get_segment_kfold_loaders(config, k=5):
    """
    Returns train and validation loaders for K-Fold using TicNonTicSegmentDataset.
    Training loader uses VideoNegFirstSampler: for each epoch, videos are shuffled; within each video all
    negative segments are yielded before positive segments so ranking cache always sees negatives first.
    """
    full_dataset = TicNonTicSegmentDataset(
        root=config['segment_root'],
        target_len=config['target_len'],
        mode='train',
        blend_dim=config.get('blend_dim', 52),
        use_blend=config.get('use_blend', True),
        blend_folder_candidates=tuple(config.get('blend_folder_candidates', ['blendshapes', 'landmarks'])),
        blend_exts=tuple(config.get('blend_exts', ['.npy', '.txt', '.csv'])),
        strict_blend=config.get('strict_blend', False),
    )

    labels = [full_dataset[i][2].item() for i in range(len(full_dataset))]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=config.get('seed', 42))

    class VideoNegFirstSampler(Sampler):
        """Sampler implementing strategy D.
        Operates over a Subset. Yields subset indices (0..len(subset)-1).
        For each epoch (iteration), shuffle video order, and inside each video yield negatives then positives.
        """
        def __init__(self, subset, seed=42):
            self.subset = subset  # torch.utils.data.Subset
            self.seed = seed
            self.epoch = 0
            base_dataset = subset.dataset  # original TicNonTicSegmentDataset
            # Build mapping: video_id_int -> {'neg': [subset_idx], 'pos': [subset_idx]}
            self.video_to_indices = {}
            for local_idx, orig_idx in enumerate(subset.indices):
                seg_dir, label = base_dataset.samples[orig_idx]
                video_id = os.path.basename(os.path.dirname(seg_dir))
                try:
                    vid_int = int(video_id)
                except ValueError:
                    vid_int = abs(hash(video_id)) % (10**6)
                bucket = self.video_to_indices.setdefault(vid_int, {'neg': [], 'pos': []})
                (bucket['pos'] if label == 1 else bucket['neg']).append(local_idx)
        def set_epoch(self, epoch):
            self.epoch = epoch
        def __iter__(self):
            rng = random.Random(self.seed + self.epoch)
            vids = list(self.video_to_indices.keys())
            rng.shuffle(vids)
            for vid in vids:
                neg_list = self.video_to_indices[vid]['neg']
                pos_list = self.video_to_indices[vid]['pos']
                rng.shuffle(neg_list)
                rng.shuffle(pos_list)
                for i in neg_list:
                    yield i
                for i in pos_list:
                    yield i
        def __len__(self):
            return len(self.subset)

    folds = []
    for train_idx, val_idx in skf.split(np.zeros(len(labels)), labels):
        train_base = TicNonTicSegmentDataset(
            root=config['segment_root'],
            target_len=config['target_len'],
            mode='train',
            blend_dim=config.get('blend_dim', 52),
            use_blend=config.get('use_blend', True),
            blend_folder_candidates=tuple(config.get('blend_folder_candidates', ['blendshapes', 'landmarks'])),
            blend_exts=tuple(config.get('blend_exts', ['.npy', '.txt', '.csv'])),
            strict_blend=config.get('strict_blend', False),
        )
        val_base = TicNonTicSegmentDataset(
            root=config['segment_root'],
            target_len=config['target_len'],
            mode='val',
            blend_dim=config.get('blend_dim', 52),
            use_blend=config.get('use_blend', True),
            blend_folder_candidates=tuple(config.get('blend_folder_candidates', ['blendshapes', 'landmarks'])),
            blend_exts=tuple(config.get('blend_exts', ['.npy', '.txt', '.csv'])),
            strict_blend=config.get('strict_blend', False),
        )
        train_subset = Subset(train_base, train_idx)
        val_subset = Subset(val_base, val_idx)
        train_sampler = VideoNegFirstSampler(train_subset, seed=config.get('seed', 42))
        train_loader = DataLoader(train_subset, batch_size=config['batch_size'], sampler=train_sampler, shuffle=False, num_workers=config.get('num_workers', 4))
        val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False, num_workers=config.get('num_workers', 4))
        folds.append((train_loader, val_loader))

    return folds

def train(model, loader, optimizer, criterion, device, cfg, epoch=1):
    model.train()
    total_ce = 0.0
    total_rank = 0.0
    total_smooth = 0.0
    total_total_loss = 0.0
    pos_rank_cases = 0
    total_pos_segments = 0
    all_preds = []  # Store predictions for accuracy calculation
    all_labels = []  # Store labels for accuracy calculation
    all_scores = []  # Store segment scores for threshold analysis
    
    # Cross-batch negative cache: video_id -> list[neg_score_tensor (detached)]
    neg_cache = defaultdict(list)
    max_cache = int(cfg.get('rank_cache_max_neg_per_video', 64))
    target_neg = max(1, int(cfg.get('rank_target_neg_per_pos', 4)))
    margin = cfg.get('rank_margin', 0.5)
    pbar = tqdm(loader, desc='Training', leave=False)
    logger = logging.getLogger('TwoStream_Logger')
    matched_video_ids = set()
    unmatched_video_ids = set()
    
    # Debug counters for accuracy calculation
    total_samples = 0
    correct_predictions = 0
    
    # Check if we're in warmup period for threshold analysis
    warmup_epochs = cfg.get('threshold_warmup_epochs', 10)
    is_warmup = epoch <= warmup_epochs
    
    for batch in pbar:
        frame_tensor, blend_tensor, labels, video_ids = batch
        frame_tensor = frame_tensor.to(device)
        blend_tensor = blend_tensor.to(device)
        labels = labels.to(device)
        video_ids = video_ids.to(device)
        optimizer.zero_grad()
        # Forward pass
        if cfg.get('model_variant', 'late_fusion') == 'late_fusion':
            frame_logits, per_frame_probs = model(frame_tensor, blend_tensor, return_frame_logits=True)
            # Use per-frame logits averaged in logit-space to form segment logits
            segment_logits_from_frames = frame_logits.mean(dim=1)  # [B, num_classes]
            ce_loss = criterion(segment_logits_from_frames, labels)
            frame_probs = per_frame_probs
        else:
            out = model(frame_tensor, blend_tensor)
            # Average per-stream logits in logit-space, then average across time for segment logits
            per_frame_logits_vit = out['per_frame_logits_vit']  # [B,T,C]
            per_frame_logits_blend = out['per_frame_logits_blend']
            per_frame_logits_avg = (per_frame_logits_vit + per_frame_logits_blend) / 2.0  # [B,T,C]
            segment_logits_from_frames = per_frame_logits_avg.mean(dim=1)  # [B,C]
            ce_loss = criterion(segment_logits_from_frames, labels)
            # Use model's averaged per-frame probabilities for ranking/aggregation
            frame_probs = out['per_frame_probs_avg']
        # Compute segment scores for this batch
        seg_scores = torch.stack([
            score_segment(frame_probs[i], window=cfg['rank_window'], alpha=cfg['rank_alpha'])
            for i in range(frame_probs.shape[0])
        ])  # [B]
        
        # store
        all_scores.extend(seg_scores.detach().cpu().numpy())
        
        # Calculate predictions and accuracy
        preds = (seg_scores > cfg.get('agg_threshold', 0.5)).long()
        batch_correct = (preds == labels).sum().item()
        batch_total = labels.size(0)
        total_samples += batch_total
        correct_predictions += batch_correct
        
        # Debug batch-level accuracy occasionally
        if cfg.get('debug_accuracy', False) and random.random() < 0.05:  # Debug 5% of batches
            batch_acc = batch_correct / batch_total if batch_total > 0 else 0
            logger.info(f"[DEBUG TRAIN] Batch acc: {batch_acc:.4f} | Labels: {labels.cpu().numpy()} | " 
                       f"Preds: {preds.cpu().numpy()} | Scores: {seg_scores.cpu().numpy().round(3)}")
        
        # Store for epoch accuracy calculation
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        rank_loss = torch.tensor(0.0, device=device)
        if cfg.get('rank_use', False):
            per_pos_losses = []
            # update total positives count
            total_pos_segments += int((labels == 1).sum().item())
            # 1) Update negative cache with current batch negatives first
            for i in range(seg_scores.shape[0 ]):
                if labels[i] == 0:
                    vid = int(video_ids[i].item())
                    neg_cache[vid].append(seg_scores[i].detach())
                    if len(neg_cache[vid]) > max_cache:
                        neg_cache[vid] = neg_cache[vid][-max_cache:]
            # 2) Anchor positives
            for i in range(seg_scores.shape[0]):
                if labels[i] != 1:
                    continue
                vid = int(video_ids[i].item())
                cached_negs = neg_cache.get(vid, [])
                if len(cached_negs) == 0:
                    unmatched_video_ids.add(vid)
                    continue  # cannot form pair now
                matched_video_ids.add(vid)
                neg_scores_tensor = torch.stack(cached_negs).to(device)
                if neg_scores_tensor.shape[0] < target_neg:
                    repeat_factor = (target_neg + neg_scores_tensor.shape[0] - 1) // neg_scores_tensor.shape[0]
                    neg_scores_tensor = neg_scores_tensor.repeat(repeat_factor)[:target_neg]
                elif neg_scores_tensor.shape[0] > target_neg:
                    idx = torch.randperm(neg_scores_tensor.shape[0], device=device)[:target_neg]
                    neg_scores_tensor = neg_scores_tensor[idx]
                pos_score = seg_scores[i].view(1)
                loss_i = pairwise_ranking_loss(pos_score, neg_scores_tensor, margin=margin, reduction='mean')
                per_pos_losses.append(loss_i)
            if per_pos_losses:
                rank_loss = torch.stack(per_pos_losses).mean()
                formed_pairs = len(per_pos_losses)
                pos_rank_cases += formed_pairs
                total_rank += rank_loss.item()
        total_ce += ce_loss.item()
        # Compute temporal smoothness loss only inside the local window around the peak
        # This matches the local_avg window used by score_segment and avoids smoothing away short peaks.
        smooth_loss = torch.tensor(0.0, device=device)
        total_diffs = 0
        if frame_probs is not None and frame_probs.dim() == 2 and frame_probs.shape[1] > 1:
            B, T = frame_probs.shape
            w = max(int(cfg.get('rank_window', 5)), 1)
            if w % 2 == 0:
                w += 1
            half = w // 2
            for i in range(B):
                seq = frame_probs[i]
                peak_idx = int(torch.argmax(seq).item())
                s = max(0, peak_idx - half)
                e = min(T, peak_idx + half + 1)
                L = e - s
                if L > 1:
                    local = seq[s:e]
                    diffs_local = local[1:] - local[:-1]
                    smooth_loss = smooth_loss + (diffs_local * diffs_local).sum()
                    total_diffs += (L - 1)
        # Normalize by total number of diffs to keep scale independent of batch size/window
        if total_diffs > 0:
            smooth_loss = smooth_loss / float(total_diffs)
        total_smooth += smooth_loss.item()  # Track total smooth loss

        # Combine losses: CE kept (scaled by ce_lambda), rank loss contributes fully (no rank_lambda),
        # smoothness weighted by smooth_lambda.
        ce_w = float(cfg.get('ce_lambda', 1.0))
        smooth_w = float(cfg.get('smooth_lambda', 0.0))
        if cfg.get('rank_use', False):
            total_loss = ce_w * ce_loss + rank_loss + smooth_w * smooth_loss
            # Debug logging for loss values
            if cfg.get('debug_loss', False) and random.random() < 0.01:  # Log 1% of batches
                logger.info(f"[DEBUG] Training loss components: ce={ce_loss.item():.6f}, rank={rank_loss.item():.6f}, "
                           f"smooth={smooth_loss.item():.6f}, ce_w={ce_w}, smooth_w={smooth_w}, "
                           f"total={total_loss.item():.6f}, expected={(ce_w * ce_loss + rank_loss + smooth_w * smooth_loss).item():.6f}")
        else:
            total_loss = ce_w * ce_loss + smooth_w * smooth_loss
        
        # Track total combined loss
        total_total_loss += total_loss.item()
            
        total_loss.backward()
        optimizer.step()
        # progress bar postfix (cumulative pair rate)
        if cfg.get('rank_use', False):
            pair_rate = (pos_rank_cases / max(1, total_pos_segments))
            pbar.set_postfix({
                'ce': f"{ce_loss.item():.3f}",
                'rank': f"{rank_loss.item():.3f}",
                'smooth': f"{smooth_loss.item():.3f}",
                'pair_rate': f"{pair_rate*100:.1f}%"
            })
    n = len(loader)
    # Calculate training accuracy
    train_acc = accuracy_score(all_labels, all_preds) if all_preds else 0.0
    direct_acc = correct_predictions / total_samples if total_samples > 0 else 0.0
    
    if cfg.get('debug_accuracy', False):
        logger.info(f"[DEBUG TRAIN] Epoch accuracy stats: sklearn={train_acc:.4f}, direct={direct_acc:.4f}, "
                   f"pos_ratio={sum(all_labels)/len(all_labels):.4f} (total={len(all_labels)})")
    
    # Threshold sensitivity analysis on training set
    # NEW: Determine best threshold from training set - SKIP during warmup
    best_train_threshold = cfg.get('agg_threshold', 0.5)  # Default
    best_train_acc = train_acc  # Default accuracy

    if cfg.get('threshold_analysis', False) and not is_warmup:
        # Generate threshold range based on configuration
        threshold_start = cfg.get('threshold_start', 0.1)
        threshold_end = cfg.get('threshold_end', 0.9)
        threshold_step = cfg.get('threshold_step', 0.1)
        thresholds = np.round(np.arange(threshold_start, threshold_end + 1e-6, threshold_step), 2)

        
        logger.info("\nTrain Set Threshold Sensitivity Analysis:")
        logger.info("Threshold\tAccuracy\tPrecision\tRecall\tF1 Score\tPositive Ratio")
        
        all_scores_np = np.array(all_scores)
        all_labels_np = np.array(all_labels)
        
        # Track metrics for each threshold to find the best ones
        threshold_metrics = []
        
        for threshold in thresholds:
            preds_at_threshold = (all_scores_np > threshold).astype(int)
            acc = accuracy_score(all_labels_np, preds_at_threshold)
            prec = precision_score(all_labels_np, preds_at_threshold, average='binary', zero_division=0)
            rec = recall_score(all_labels_np, preds_at_threshold, average='binary', zero_division=0)
            f1 = f1_score(all_labels_np, preds_at_threshold, average='binary', zero_division=0)
            pos_ratio = preds_at_threshold.mean()
            
            # Store metrics for finding optimal threshold
            threshold_metrics.append({
                'threshold': threshold,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'pos_ratio': pos_ratio
            })
            
            # Display with two decimal places for threshold
            logger.info(f"{threshold:.2f}\t{acc:.4f}\t{prec:.4f}\t{rec:.4f}\t{f1:.4f}\t{pos_ratio:.4f}")
        
        # Find best thresholds based on different prioritization criteria
        # 1. Best accuracy threshold: Sort by accuracy (primary), recall (secondary), precision (tertiary), and f1 (quaternary)
        best_acc_threshold = sorted(threshold_metrics, 
                                  key=lambda x: (x['accuracy'], x['recall'], x['precision'], x['f1']), 
                                  reverse=True)[0]['threshold']
        
        # 2. Best recall threshold: Sort by recall (primary), accuracy (secondary), precision (tertiary), and f1 (quaternary)
        best_recall_threshold = sorted(threshold_metrics, 
                                     key=lambda x: (x['recall'], x['accuracy'], x['precision'], x['f1']), 
                                     reverse=True)[0]['threshold']
        
        # Log best thresholds without showing the accuracy values
        logger.info(f"Best threshold by accuracy: {best_acc_threshold:.2f}")
        logger.info(f"Best threshold by recall: {best_recall_threshold:.2f}")
        
        # Fine-grained search around best threshold on TRAINING SET
        if cfg.get('threshold_fine_search', False):
            fine_step = cfg.get('threshold_step', 0.1) * cfg.get('threshold_fine_multiplier', 0.2)
            # Create fine-grained thresholds around best threshold
            fine_start = max(0, best_acc_threshold - cfg.get('threshold_step', 0.1))
            fine_end = min(1, best_acc_threshold + cfg.get('threshold_step', 0.1))
            fine_thresholds = np.round(np.arange(fine_start, fine_end + 1e-6, fine_step), 3)
            
            logger.info("\nFine-Grained Threshold Sensitivity Analysis:")
            logger.info("Threshold\tAccuracy\tPrecision\tRecall\tF1 Score\tPositive Ratio")
            
            fine_threshold_metrics = []
            
            for threshold in fine_thresholds:
                # Skip the exact threshold we already tested
                if abs(threshold - best_acc_threshold) < 1e-6:
                    continue
                    
                preds_at_threshold = (all_scores_np > threshold).astype(int)
                acc = accuracy_score(all_labels_np, preds_at_threshold)
                prec = precision_score(all_labels_np, preds_at_threshold, average='binary', zero_division=0)
                rec = recall_score(all_labels_np, preds_at_threshold, average='binary', zero_division=0)
                f1 = f1_score(all_labels_np, preds_at_threshold, average='binary', zero_division=0)
                pos_ratio = preds_at_threshold.mean()
                
                # Store metrics for finding optimal threshold
                fine_threshold_metrics.append({
                    'threshold': threshold,
                    'accuracy': acc,
                    'precision': prec,
                    'recall': rec,
                    'f1': f1,
                    'pos_ratio': pos_ratio
                })
                
                # Display with three decimal places for fine-grained thresholds
                logger.info(f"{threshold:.3f}\t{acc:.4f}\t{prec:.4f}\t{rec:.4f}\t{f1:.4f}\t{pos_ratio:.4f}")
            
            # Combine original and fine-grained metrics for final selection
            all_threshold_metrics = threshold_metrics + fine_threshold_metrics
            
            # Find best accuracy threshold: Sort by accuracy (primary), recall (secondary), precision (tertiary), and f1 (quaternary)
            best_overall_metric = sorted(all_threshold_metrics, 
                                      key=lambda x: (x['accuracy'], x['recall'], x['precision'], x['f1']), 
                                      reverse=True)[0]
            
            # Update best values if fine-grained search found better threshold
            if best_overall_metric['accuracy'] > best_train_acc:
                best_train_threshold = best_overall_metric['threshold']
                best_train_acc = best_overall_metric['accuracy']
                logger.info(f"Fine-grained search found better threshold: {best_train_threshold:.3f} with accuracy: {best_train_acc:.4f}")
            else:
                logger.info(f"Fine-grained search did not improve best threshold ({best_train_threshold:.2f})")
    elif is_warmup:
        logger.info(f"[WARMUP] Skipping ALL threshold sensitivity analysis (epoch {epoch}/{warmup_epochs})")
    
    avg_ce = total_ce / n
    avg_rank_per_batch = total_rank / n  # for reporting consistency
    avg_smooth = total_smooth / n
    avg_total = total_total_loss / n
    
    # Sanity check for total loss calculation
    expected_total = ce_w * avg_ce + avg_rank_per_batch + smooth_w * avg_smooth
    if abs(avg_total - expected_total) > 0.001:  # tolerance
        logger.warning(f"[LOSS MISMATCH] avg_total={avg_total:.6f} != expected={expected_total:.6f} "
                      f"(diff={avg_total - expected_total:.6f})")
    
    pair_rate_epoch = (pos_rank_cases / max(1, total_pos_segments)) if cfg.get('rank_use', False) else 0.0
    if cfg.get('rank_use', False) and cfg.get('rank_log_pair_videos', False):
        # derive sets difference to avoid overlap in case both happened
        only_unmatched = sorted(unmatched_video_ids - matched_video_ids)
        matched_sorted = sorted(matched_video_ids)
        logger.info(f"[Train] Matched positive video IDs ({len(matched_sorted)}): {matched_sorted}")
        logger.info(f"[Train] Unmatched positive video IDs ({len(only_unmatched)}): {only_unmatched}")
    # return average losses and training accuracy
    return avg_total, avg_ce, avg_rank_per_batch, avg_smooth, pair_rate_epoch, best_train_acc, best_train_threshold

def validate(model, loader, criterion, device, cfg, epoch=1):
    model.eval()
    total_ce = 0.0
    total_rank = 0.0
    total_smooth = 0.0
    total_total_loss = 0.0
    pos_rank_cases = 0
    total_pos_segments = 0
    all_preds, all_labels = [], []
    all_scores = []  # 存储所有样本的分数用于阈值分析
    all_probs = []  # 存储所有样本的类别概率
    pbar = tqdm(loader, desc='Validating', leave=False)
    logger = logging.getLogger('TwoStream_Logger')
    matched_video_ids = set()
    unmatched_video_ids = set()
    
    # Debug counters for validation accuracy
    total_samples = 0
    correct_predictions = 0
    
    # Initialize negative cache similar to training function
    neg_cache = defaultdict(list)
    max_cache = int(cfg.get('rank_cache_max_neg_per_video', 64))
    target_neg = max(1, int(cfg.get('rank_target_neg_per_pos', 4)))
    margin = cfg.get('rank_margin', 0.5)
    
    # Check if we're in warmup period for threshold analysis
    warmup_epochs = cfg.get('threshold_warmup_epochs', 10)
    is_warmup = epoch <= warmup_epochs
    
    with torch.no_grad():
        for batch in pbar:
            frame_tensor, blend_tensor, labels, video_ids = batch
            frame_tensor = frame_tensor.to(device)
            blend_tensor = blend_tensor.to(device)
            labels = labels.to(device)
            video_ids = video_ids.to(device)
            if cfg.get('model_variant', 'late_fusion') == 'late_fusion':
                frame_logits, per_frame_probs = model(frame_tensor, blend_tensor, return_frame_logits=True)
                # compute CE from per-frame logits averaged in logit space
                segment_logits_from_frames = frame_logits.mean(dim=1)
                ce_loss = criterion(segment_logits_from_frames, labels)
                frame_probs = per_frame_probs
                agg_scores = torch.stack([
                    score_segment(frame_probs[i], window=cfg['rank_window'], alpha=cfg['rank_alpha'])
                    for i in range(frame_probs.shape[0])
                ])
                # store scores for threshold sensitivity analysis
                all_scores.extend(agg_scores.cpu().numpy())
                
                preds = (agg_scores > cfg.get('agg_threshold', 0.5)).long()
                seg_probs = torch.stack([1.0 - agg_scores, agg_scores], dim=1)
            else:
                out = model(frame_tensor, blend_tensor)
                # compute CE from per-frame logits averaged across streams (logit-space) and time
                per_frame_logits_vit = out['per_frame_logits_vit']
                per_frame_logits_blend = out['per_frame_logits_blend']
                per_frame_logits_avg = (per_frame_logits_vit + per_frame_logits_blend) / 2.0
                segment_logits_from_frames = per_frame_logits_avg.mean(dim=1)
                ce_loss = criterion(segment_logits_from_frames, labels)
                # use model's per-frame averaged probabilities for aggregation/ranking
                frame_probs = out['per_frame_probs_avg']
                agg_scores = torch.stack([
                    score_segment(frame_probs[i], window=cfg['rank_window'], alpha=cfg['rank_alpha'])
                    for i in range(frame_probs.shape[0])
                ])
                # store scores for threshold sensitivity analysis
                all_scores.extend(agg_scores.cpu().numpy())
                
                preds = (agg_scores > cfg.get('agg_threshold', 0.5)).long()
                seg_probs = torch.stack([1.0 - agg_scores, agg_scores], dim=1)
            
            # Calculate batch accuracy metrics
            batch_correct = (preds == labels).sum().item()
            batch_total = labels.size(0)
            total_samples += batch_total
            correct_predictions += batch_correct
            
            # Debug batch-level accuracy occasionally
            if cfg.get('debug_accuracy', False) and random.random() < 0.05:  # Debug 5% of batches
                batch_acc = batch_correct / batch_total if batch_total > 0 else 0
                logger.info(f"[DEBUG VAL] Batch acc: {batch_acc:.4f} | Labels: {labels.cpu().numpy()} | " 
                           f"Preds: {preds.cpu().numpy()} | Scores: {agg_scores.cpu().numpy().round(3)}")
                           
            total_ce += ce_loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(seg_probs.cpu().numpy())
            
            if cfg.get('rank_use', False):
                total_pos_segments += int((labels == 1).sum().item())
                seg_scores = agg_scores
                
                # Update negative cache with current batch negatives first (similar to training)
                for i in range(seg_scores.shape[0]):
                    if labels[i] == 0:
                        vid = int(video_ids[i].item())
                        neg_cache[vid].append(seg_scores[i].detach())
                        if len(neg_cache[vid]) > max_cache:
                            neg_cache[vid] = neg_cache[vid][-max_cache:]
                
                # Anchor positives and match with cached negatives (similar to training)
                per_pos_losses = []
                for i in range(seg_scores.shape[0]):
                    if labels[i] != 1:
                        continue
                    vid = int(video_ids[i].item())
                    cached_negs = neg_cache.get(vid, [])
                    if len(cached_negs) == 0:
                        unmatched_video_ids.add(vid)
                        continue  # cannot form pair now
                    matched_video_ids.add(vid)
                    neg_scores_tensor = torch.stack(cached_negs).to(device)
                    if neg_scores_tensor.shape[0] < target_neg:
                        repeat_factor = (target_neg + neg_scores_tensor.shape[0] - 1) // neg_scores_tensor.shape[0]
                        neg_scores_tensor = neg_scores_tensor.repeat(repeat_factor)[:target_neg]
                    elif neg_scores_tensor.shape[0] > target_neg:
                        idx = torch.randperm(neg_scores_tensor.shape[0], device=device)[:target_neg]
                        neg_scores_tensor = neg_scores_tensor[idx]
                    pos_score = seg_scores[i].view(1)
                    loss_i = pairwise_ranking_loss(pos_score, neg_scores_tensor, margin=margin, reduction='mean')
                    per_pos_losses.append(loss_i)
                
                if per_pos_losses:
                    batch_rank = torch.stack(per_pos_losses).mean()
                    total_rank += batch_rank.item()
                    pos_rank_cases += len(per_pos_losses)
                    rank_loss_tensor = batch_rank
                else:
                    rank_loss_tensor = torch.tensor(0.0, device=device)
            else:
                rank_loss_tensor = torch.tensor(0.0, device=device)

            # Compute smooth loss inside peak window (same as in training) and normalize by number of diffs
            smooth_loss = torch.tensor(0.0, device=device)
            total_diffs = 0
            if frame_probs is not None and frame_probs.dim() == 2 and frame_probs.shape[1] > 1:
                B, T = frame_probs.shape
                w = max(int(cfg.get('rank_window', 5)), 1)
                if w % 2 == 0:
                    w += 1
                half = w // 2
                for i in range(B):
                    seq = frame_probs[i]
                    peak_idx = int(torch.argmax(seq).item())
                    s = max(0, peak_idx - half)
                    e = min(T, peak_idx + half + 1)
                    L = e - s
                    if L > 1:
                        local = seq[s:e]
                        diffs_local = local[1:] - local[:-1]
                        smooth_loss = smooth_loss + (diffs_local * diffs_local).sum()
                        total_diffs += (L - 1)
            if total_diffs > 0:
                smooth_loss = smooth_loss / float(total_diffs)
            
            # Track total smooth loss
            total_smooth += smooth_loss.item()

            # Combine batch total loss using cfg weights (CE scaled by ce_lambda)
            ce_w = float(cfg.get('ce_lambda', 1.0))
            smooth_w = float(cfg.get('smooth_lambda', 0.0))
            if cfg.get('rank_use', False):
                print("using rank loss in val")
                batch_total_loss = ce_w * ce_loss + rank_loss_tensor + smooth_w * smooth_loss
                # Debug logging for validation loss components
                if cfg.get('debug_loss', False) and random.random() < 0.01:  # Log 1% of batches
                    logger.info(f"[DEBUG] Validation loss components: ce={ce_loss.item():.6f}, rank={rank_loss_tensor.item():.6f}, "
                              f"smooth={smooth_loss.item():.6f}, ce_w={ce_w}, smooth_w={smooth_w}, "
                              f"total={batch_total_loss.item():.6f}, expected={(ce_w * ce_loss + rank_loss_tensor + smooth_w * smooth_loss).item():.6f}")
            else:
                batch_total_loss = ce_w * ce_loss + smooth_w * smooth_loss

            # Accumulate total loss scalar for averaging
            total_total_loss += batch_total_loss.item()
    
    # Calculate validation accuracy using two methods for comparison
    val_acc = accuracy_score(all_labels, all_preds)
    direct_acc = correct_predictions / total_samples if total_samples > 0 else 0.0
    
    if cfg.get('debug_accuracy', False):
        logger.info(f"[DEBUG VAL] Epoch accuracy stats: sklearn={val_acc:.4f}, direct={direct_acc:.4f}, "
                   f"pos_ratio={sum(all_labels)/len(all_labels):.4f} (total={len(all_labels)})")
    
    # Add threshold sensitivity analysis for validation set - SKIP during warmup
    best_acc_at_any_threshold = val_acc  # Default to current threshold accuracy
    best_threshold = cfg.get('agg_threshold', 0.5)  # Default threshold
    
    if cfg.get('threshold_analysis', False) and not is_warmup:
        # Generate threshold range based on configuration
        threshold_start = cfg.get('threshold_start', 0.1)
        threshold_end = cfg.get('threshold_end', 0.9)
        threshold_step = cfg.get('threshold_step', 0.1)
        thresholds = np.round(np.arange(threshold_start, threshold_end + 1e-6, threshold_step), 2)
        
        logger.info("\nValidation Set Threshold Sensitivity Analysis:")
        logger.info("Threshold\tAccuracy\tPrecision\tRecall\tF1 Score\tPositive Ratio")
        
        all_scores_np = np.array(all_scores)
        all_labels_np = np.array(all_labels)
        
        # Track metrics for each threshold to find the best one
        threshold_metrics = []
        
        for threshold in thresholds:
            preds_at_threshold = (all_scores_np > threshold).astype(int)
            acc = accuracy_score(all_labels_np, preds_at_threshold)
            prec = precision_score(all_labels_np, preds_at_threshold, average='binary', zero_division=0)
            rec = recall_score(all_labels_np, preds_at_threshold, average='binary', zero_division=0)
            f1 = f1_score(all_labels_np, preds_at_threshold, average='binary', zero_division=0)
            pos_ratio = preds_at_threshold.mean()
            
            threshold_metrics.append({
                'threshold': threshold,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'pos_ratio': pos_ratio
            })
            
            logger.info(f"{threshold:.2f}\t{acc:.4f}\t{prec:.4f}\t{rec:.4f}\t{f1:.4f}\t{pos_ratio:.4f}")
        
        # Find best accuracy threshold
        best_threshold_metric = sorted(threshold_metrics, 
                                  key=lambda x: (x['accuracy'], x['recall'], x['precision'], x['f1']), 
                                  reverse=True)[0]
        
        best_acc_at_any_threshold = best_threshold_metric['accuracy']
        best_threshold = best_threshold_metric['threshold']
        
        logger.info(f"Best threshold by accuracy: {best_threshold:.2f}")
        logger.info(f"Best recall threshold: {sorted(threshold_metrics, key=lambda x: (x['recall'], x['accuracy'], x['precision'], x['f1']), reverse=True)[0]['threshold']:.2f}")
        
        # NOTE: Fine-grained search REMOVED from validation function
        # Only training keeps fine-grained search for checkpoint threshold
        
    elif is_warmup:
        logger.info(f"[WARMUP] Skipping validation threshold analysis (epoch {epoch}/{warmup_epochs})")
    
    avg_ce = total_ce / len(loader)
    avg_rank = (total_rank / max(1, pos_rank_cases)) if cfg.get('rank_use', False) else 0.0
    avg_smooth = total_smooth / len(loader)
    pair_rate_epoch = (pos_rank_cases / max(1, total_pos_segments)) if cfg.get('rank_use', False) else 0.0
    if cfg.get('rank_use', False) and cfg.get('rank_log_pair_videos', False):
        only_unmatched = sorted(unmatched_video_ids - matched_video_ids)
        matched_sorted = sorted(matched_video_ids)
        logger.info(f"[Val] Matched positive video IDs ({len(matched_sorted)}): {matched_sorted}")
        logger.info(f"[Val] Unmatched positive video IDs ({len(only_unmatched)}): {only_unmatched}")
    all_probs_np = np.concatenate(all_probs, axis=0)
    avg_total = total_total_loss / len(loader)
    return (avg_total, avg_ce, avg_rank, avg_smooth, pair_rate_epoch, best_acc_at_any_threshold, best_threshold), all_preds, all_labels, all_probs_np

def setup_logger(log_path):
    logger = logging.getLogger('TwoStream_Logger')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def compute_class_weights(labels, num_classes):
    """
    Compute class weights based on the frequency of each class.
    :param labels: List of all labels in the dataset.
    :param num_classes: Total number of classes.
    :return: A tensor of class weights.
    """
    class_counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    class_weights = total_samples / (num_classes * class_counts)
    return torch.tensor(class_weights, dtype=torch.float)

def build_infer_config(cfg: dict):
    """Extract minimal fields required for rebuilding model at inference time."""
    out = {}
    for k in INFER_CONFIG_KEYS:
        if k in cfg:
            out[k] = cfg[k]
    return out

def main(config, start_fold=0):
    """
    Main function for training with K-Fold cross-validation.
    :param config: Configuration dictionary.
    :param start_fold: The fold index to start training from (default is 0).
    """
    results_path = os.path.join(config['results_dir'], config['exp'])
    os.makedirs(results_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_path = os.path.join(results_path, 'training.log')
    logger = setup_logger(log_path)
    logger.info("Starting K-Fold training with configuration: %s", config)

    # Set random seed for reproducibility
    set_random_seeds(config['seed'])

    # Initialize ViT encoder with (optional) LoRA. Dataset no longer normalizes images; feature_extractor handles normalization.
    vit_encoder = ViTFrameEncoder(
        model_name=config['vit_model_name'],
        device=device,
        dropout=config['vit_dropout'],
        use_lora=config.get('vit_use_lora', False),
        lora_r=config.get('vit_lora_r', 8),
        lora_alpha=config.get('vit_lora_alpha', 32),
        lora_dropout=config.get('vit_lora_dropout', 0.1),
        lora_target_modules=config.get('vit_lora_targets', ['query', 'value'])
    )
    config['vit_tcn_config']['num_inputs'] = vit_encoder.hidden_dim  # Dynamically set num_inputs
    # sync blend stream input dim with dataset feature dim
    config['blend_tcn_config']['num_inputs'] = int(config.get('blend_dim', 52))

    # Always use segment loaders
    folds = get_segment_kfold_loaders(config, k=5)

    all_fold_metrics = []

    for fold_idx, (train_loader, val_loader) in enumerate(folds):
        # Skip folds before the start_fold
        if fold_idx < start_fold:
            logger.info(f"Skipping Fold {fold_idx + 1} as it is before the start_fold {start_fold + 1}")
            continue

        logger.info(f"Starting Fold {fold_idx + 1}")
        model = LateFusionModel(
            vit_encoder=vit_encoder,
            vit_tcn_config=config['vit_tcn_config'],
            blend_tcn_config=config['blend_tcn_config'],
            num_classes=config['num_classes'],
            fusion_mode=config['fusion_mode']
        ).to(device) if config.get('model_variant','late_fusion')=='late_fusion' else FrameProbAvgFusionModel(
            vit_encoder=vit_encoder,
            vit_tcn_config=config['vit_tcn_config'],
            blend_tcn_config=config['blend_tcn_config'],
            num_classes=config['num_classes']
        ).to(device)
        # Ensure encoder in train mode if LoRA is used so LoRA layers update
        if config.get('vit_use_lora', False):
            model.vit_encoder.train()

        # Compute class weights from current fold's training set
        train_labels = [train_loader.dataset[i][2].item() for i in range(len(train_loader.dataset))]
        class_weights = compute_class_weights(train_labels, config['num_classes']).to(device)

        # Pass class weights to CrossEntropyLoss
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        
        # Checkpoint paths
        checkpoint_path = os.path.join(results_path, f'checkpoint_fold_{fold_idx + 1}.pth')
        best_model_path = os.path.join(results_path, f'best_model_fold_{fold_idx + 1}.pth')
        best_model_by_acc_path = os.path.join(results_path, f'best_model_by_acc_fold_{fold_idx + 1}.pth')

        # Load checkpoint if exists
        start_epoch = 1
        best_loss = float('inf')  # Initialize with infinity for loss minimization
        best_acc = 0.0  # Track best accuracy for model saving
        best_threshold = config.get('agg_threshold', 0.5)  # Best threshold for accuracy
        
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint for Fold {fold_idx + 1} from {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_loss = checkpoint.get('best_loss', float('inf'))
                best_acc = checkpoint.get('best_acc', 0.0)
                best_threshold = checkpoint.get('best_threshold', config.get('agg_threshold', 0.5))
                logger.info(f"Resuming training from epoch {start_epoch} with best loss {best_loss:.4f} and best accuracy {best_acc:.4f}")
            except (RuntimeError, EOFError, pickle.PickleError) as e:
                logger.warning(f"Checkpoint file {checkpoint_path} is corrupted: {e}")
                logger.info("Removing corrupted checkpoint and starting fresh training")
                os.remove(checkpoint_path)
                # Reset to default values
                start_epoch = 1
                best_loss = float('inf')
                best_acc = 0.0
                best_threshold = config.get('agg_threshold', 0.5)

        epoch_pbar = tqdm(range(start_epoch, config['epochs'] + 1), desc=f'Fold {fold_idx + 1} Epochs')
        no_improve_epochs = 0  # early stopping counter based on validation loss
        best_model_state = None  # Store best model state based on validation loss
        best_model_by_acc_state = None  # Store best model state based on accuracy
        
        for epoch in epoch_pbar:
            # For each epoch, set the sampler's epoch so it reshuffles videos
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
                
            # Train
            train_total, train_ce, train_rank, train_smooth, train_pair_rate, train_acc, best_train_threshold = train(model, train_loader, optimizer, criterion, device, config, epoch)
            
            # Validate - now returns best accuracy at any threshold
            (val_total, val_ce, val_rank, val_smooth, val_pair_rate, val_acc_at_best_threshold, best_val_threshold), val_preds, val_labels, val_probs = validate(model, val_loader, criterion, device, config, epoch)
            
            # Log with combined metrics
            logger.info(
                f"Fold {fold_idx + 1} | Epoch {epoch} | "
                f"Train_Total {train_total:.4f} (CE:{train_ce:.4f} Rank:{train_rank:.4f} Smooth:{train_smooth:.4f}) | "
                f"Val_Total {val_total:.4f} (CE:{val_ce:.4f} Rank:{val_rank:.4f} Smooth:{val_smooth:.4f}) | "
                f"Best_Val_Acc {val_acc_at_best_threshold:.4f} @ threshold={best_val_threshold:.2f} | "
                f"PairRate {train_pair_rate*100:.1f}%/{val_pair_rate*100:.1f}%"
            )

            scheduler.step()

            # Early stopping based on validation loss (not accuracy)
            if train_total < best_loss:
                best_loss = train_total
                no_improve_epochs = 0  # reset patience counter
                best_model_state = model.state_dict()
                
                # Save best model based on validation loss
                torch.save(best_model_state, best_model_path)
                logger.info(f"Saved best model (by loss) for Fold {fold_idx + 1} at epoch {epoch} (val_loss={best_loss:.4f})")
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= config.get('early_stop_patience', 10):
                    logger.info(f"Early stopping triggered on Fold {fold_idx + 1} at epoch {epoch} (patience={config.get('early_stop_patience')}, no improvement in training loss)")
                    break
            
            # Track and save best model based on accuracy across all thresholds
            if val_acc_at_best_threshold > best_acc:
                best_acc = val_acc_at_best_threshold
                best_threshold = best_val_threshold
                best_model_by_acc_state = model.state_dict()
                
                # Save best model based on accuracy
                torch.save(best_model_by_acc_state, best_model_by_acc_path)
                
                # Also create inference package with best threshold
                best_infer_path = os.path.join(results_path, f'best_inference_fold_{fold_idx + 1}.pt')
                infer_config = build_infer_config(config)
                infer_config['agg_threshold'] = best_threshold  # Update threshold in inference config
                torch.save({
                    'model_state_dict': best_model_by_acc_state,
                    'config': infer_config,
                    'class_names': config['class_names'],
                    'best_threshold': best_threshold
                }, best_infer_path)
                
                logger.info(f"Saved best model (by accuracy) for Fold {fold_idx + 1} at epoch {epoch} (acc={best_acc:.4f}, threshold={best_threshold:.2f})")

            # Periodic checkpoint (includes optimizer + minimal infer config for resume)
            if epoch % config.get('checkpoint_interval', 5) == 0 or epoch == config['epochs']:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                    'best_acc': best_acc,
                    'best_threshold': best_threshold,
                    'inference_config': build_infer_config(config),
                    'class_names': config['class_names']
                }
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Checkpoint for Fold {fold_idx + 1} saved at epoch {epoch}")

        all_fold_metrics.append({'accuracy': best_acc, 'loss': best_loss, 'threshold': best_threshold})
        logger.info(f"Fold {fold_idx + 1} completed. Best validation accuracy: {best_acc:.4f} at threshold {best_threshold:.2f}")

    # Summarize results across all folds
    avg_accuracy = np.mean([m['accuracy'] for m in all_fold_metrics])
    avg_loss = np.mean([m['loss'] for m in all_fold_metrics])
    logger.info(f"K-Fold Cross-Validation Complete")
    logger.info(f"Average Accuracy: {avg_accuracy:.4f}")
    logger.info(f"Average Loss: {avg_loss:.4f}")
    for i, metrics in enumerate(all_fold_metrics):
        logger.info(f"Fold {i+1}: Accuracy={metrics['accuracy']:.4f}, Loss={metrics['loss']:.4f}, Threshold={metrics['threshold']:.2f}")

def test(config):
    results_path = os.path.join(config['results_dir'], config['exp'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = get_segment_kfold_loaders(config, k=5)
    all_labels, all_preds, all_probs = [], [], []

    fold_pbar = tqdm(enumerate(folds), total=len(folds), desc='Testing Folds')
    for fold_idx, (_, val_loader) in fold_pbar:
        vit_encoder = ViTFrameEncoder(
            model_name=config['vit_model_name'],
            device=device,
            dropout=config['vit_dropout'],
            use_lora=config.get('vit_use_lora', False),
            lora_r=config.get('vit_lora_r', 8),
            lora_alpha=config.get('vit_lora_alpha', 32),
            lora_dropout=config.get('vit_lora_dropout', 0.1),
            lora_target_modules=config.get('vit_lora_targets', ['query', 'value'])
        )
        model = LateFusionModel(
            vit_encoder=vit_encoder,
            vit_tcn_config=config['vit_tcn_config'],
            blend_tcn_config=config['blend_tcn_config'],
            num_classes=config['num_classes'],
            fusion_mode=config['fusion_mode']
        ).to(device) if config.get('model_variant','late_fusion')=='late_fusion' else FrameProbAvgFusionModel(
            vit_encoder=vit_encoder,
            vit_tcn_config=config['vit_tcn_config'],
            blend_tcn_config=config['blend_tcn_config'],
            num_classes=config['num_classes']
        ).to(device)
        model_path = os.path.join(results_path, f'best_model_fold_{fold_idx + 1}.pth')
        obj = torch.load(model_path, map_location=device, weights_only=False)
        # Backward compatibility: file may be raw state_dict or full package
        state_dict = obj['model_state_dict'] if isinstance(obj, dict) and 'model_state_dict' in obj else obj
        model.load_state_dict(state_dict)
        model.eval()

        # No class weights needed for testing
        (val_total, val_ce, val_rank, val_smooth, val_pair_rate, val_acc, _), val_preds, val_labels, val_probs = validate(model, val_loader, torch.nn.CrossEntropyLoss(), device, config)
        all_labels.extend(val_labels)
        all_preds.extend(val_preds)
        all_probs.append(val_probs)

    all_probs = np.concatenate(all_probs, axis=0)
    y_true_onehot = label_binarize(all_labels, classes=list(range(config['num_classes'])))

    overall_accuracy = accuracy_score(all_labels, all_preds)
    overall_precision = precision_score(all_labels, all_preds, average='weighted')
    overall_recall = recall_score(all_labels, all_preds, average='weighted')
    overall_f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"Overall Metrics:")
    print(f"Accuracy: {overall_accuracy:.4f}")
    print(f"Precision: {overall_precision:.4f}")
    print(f"Recall: {overall_recall:.4f}")
    print(f"F1 Score: {overall_f1:.4f}")
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print("\nThreshold Sensitivity Analysis:")
    print("Threshold\tAccuracy\tPrecision\tRecall\tF1 Score\tPositive Ratio")
    for threshold in thresholds:
        preds_at_threshold = (all_probs[:, 1] > threshold).astype(int)
        acc = accuracy_score(all_labels, preds_at_threshold)
        prec = precision_score(all_labels, preds_at_threshold, average='binary', zero_division=0)
        rec = recall_score(all_labels, preds_at_threshold, average='binary', zero_division=0)
        f1 = f1_score(all_labels, preds_at_threshold, average='binary', zero_division=0)
        pos_ratio = preds_at_threshold.mean()
        print(f"{threshold:.1f}\t{acc:.4f}\t{prec:.4f}\t{rec:.4f}\t{f1:.4f}\t{pos_ratio:.4f}")

    plot_confusion_matrix(all_labels, all_preds, config['class_names'], os.path.join(results_path, 'confusion_matrix.png'))
    plot_roc_curve(y_true_onehot, all_probs, config['num_classes'], os.path.join(results_path, 'roc_curve.png'))

if __name__ == "__main__":
    mode = 'train'  # Change to 'test' for testing
    if mode == 'train':
        main(config, start_fold=config['start_fold'])
    elif mode == 'test':
        test(config)
