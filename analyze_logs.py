import re
import os
# removed argparse
import matplotlib.pyplot as plt
from collections import defaultdict

# Internal config instead of argparse
ANALYZE_CONFIG = {
    'log_path': r'A:\youtube_finetune\result_analysis\training.log',
    'out_dir': r'A:\youtube_finetune\result_analysis'
}

# Regex patterns to extract epoch metrics and best model saves
EPOCH_LINE_PATTERN = re.compile(
    r"Fold (\d+) \| Epoch (\d+) \| .*?Train_Total ([0-9.]+) \(CE:([0-9.]+).*?Val_Total ([0-9.]+) \(CE:([0-9.]+).*?Best_Val_Acc ([0-9.]+) @ threshold=([0-9.]+)")
# Optional line when best accuracy model saved (to double-check)
BEST_ACC_SAVE_PATTERN = re.compile(
    r"Saved best model \(by accuracy\) for Fold (\d+) at epoch (\d+) \(acc=([0-9.]+), threshold=([0-9.]+)\)"
)

BEST_LOSS_SAVE_PATTERN = re.compile(
    r"Saved best model \(by loss\) for Fold (\d+) at epoch (\d+) \(val_loss=([0-9.]+)\)"
)


def parse_log(log_path):
    """Parse the training log and extract per-fold epoch metrics and best checkpoints.
    Returns:
        folds: dict[fold] -> {
            'epochs': [int],
            'train_ce': [float],
            'val_ce': [float],
            'train_total': [float],
            'val_total': [float],
            'val_acc': [float],
            'threshold': [float]
        }
        best_by_acc: dict[fold] -> {'epoch': int, 'acc': float, 'threshold': float}
        best_by_loss: dict[fold] -> {'epoch': int, 'val_loss': float}
    """
    folds = defaultdict(lambda: {
        'epochs': [],
        'train_ce': [],
        'val_ce': [],
        'train_total': [],
        'val_total': [],
        'val_acc': [],
        'threshold': []
    })
    best_by_acc = {}
    best_by_loss = {}

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Epoch summary line
            m = EPOCH_LINE_PATTERN.search(line)
            if m:
                fold, epoch = int(m.group(1)), int(m.group(2))
                train_total = float(m.group(3))
                train_ce = float(m.group(4))
                val_total = float(m.group(5))
                val_ce = float(m.group(6))
                val_acc = float(m.group(7))
                threshold = float(m.group(8))
                folds[fold]['epochs'].append(epoch)
                folds[fold]['train_total'].append(train_total)
                folds[fold]['train_ce'].append(train_ce)
                folds[fold]['val_total'].append(val_total)
                folds[fold]['val_ce'].append(val_ce)
                folds[fold]['val_acc'].append(val_acc)
                folds[fold]['threshold'].append(threshold)
                # Track best by accuracy (highest val_acc, tie-breaker: lower epoch)
                if fold not in best_by_acc or val_acc > best_by_acc[fold]['acc']:
                    best_by_acc[fold] = {
                        'epoch': epoch,
                        'acc': val_acc,
                        'threshold': threshold,
                        'train_ce': train_ce,
                        'val_ce': val_ce,
                        'train_total': train_total,
                        'val_total': val_total
                    }
                continue
            # Best model saved by accuracy line (optional, may duplicate info)
            m2 = BEST_ACC_SAVE_PATTERN.search(line)
            if m2:
                fold = int(m2.group(1))
                epoch = int(m2.group(2))
                acc = float(m2.group(3))
                threshold = float(m2.group(4))
                # Only update if better than currently stored
                if fold not in best_by_acc or acc > best_by_acc[fold]['acc']:
                    # CE losses will be filled from epoch metrics; skip here if absent
                    best_by_acc.setdefault(fold, {})
                    best_by_acc[fold].update({'epoch': epoch, 'acc': acc, 'threshold': threshold})
                continue
            # Best model saved by loss line
            m3 = BEST_LOSS_SAVE_PATTERN.search(line)
            if m3:
                fold = int(m3.group(1))
                epoch = int(m3.group(2))
                val_loss = float(m3.group(3))
                if fold not in best_by_loss or val_loss < best_by_loss[fold]['val_loss']:
                    best_by_loss[fold] = {'epoch': epoch, 'val_loss': val_loss}
                continue
    return folds, best_by_acc, best_by_loss


def plot_losses(fold_id, epochs, train_ce, val_ce, out_dir):
    """Plot CE training and validation loss curves for a fold."""
    if not epochs:
        return
    plt.figure(figsize=(6,4))
    plt.plot(epochs, train_ce, label='Train CE', marker='o')
    plt.plot(epochs, val_ce, label='Val CE', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('CrossEntropy Loss')
    plt.title(f'Fold {fold_id} CE Loss Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f'fold_{fold_id}_ce_loss.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def summarize(folds, best_by_acc, best_by_loss, out_dir):
    """Print and save a summary of best metrics per fold."""
    lines = []
    lines.append('Fold\tBestEpoch(Acc)\tBestAcc\tBestThreshold\tTrainCE@Best\tValCE@Best\tTrainTotal@Best\tValTotal@Best\tBestLossEpoch\tBestValLoss')
    for fold in sorted(folds.keys()):
        acc_part = best_by_acc.get(fold, {})
        loss_part = best_by_loss.get(fold, {})
        line = (
            f"{fold}\t"
            f"{acc_part.get('epoch','-')}\t"
            f"{acc_part.get('acc','-')}\t"
            f"{acc_part.get('threshold','-')}\t"
            f"{acc_part.get('train_ce','-')}\t"
            f"{acc_part.get('val_ce','-')}\t"
            f"{acc_part.get('train_total','-')}\t"
            f"{acc_part.get('val_total','-')}\t"
            f"{loss_part.get('epoch','-')}\t"
            f"{loss_part.get('val_loss','-')}"
        )
        lines.append(line)
    summary_txt = "\n".join(lines)
    print(summary_txt)
    with open(os.path.join(out_dir, 'log_analysis_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(summary_txt)


def main():
    cfg = ANALYZE_CONFIG
    log_path = cfg['log_path']
    out_dir = cfg['out_dir']
    if not os.path.isfile(log_path):
        raise FileNotFoundError(f'Log file not found: {log_path}')
    folds, best_by_acc, best_by_loss = parse_log(log_path)
    os.makedirs(out_dir, exist_ok=True)
    for fold, data in folds.items():
        plot_losses(fold, data['epochs'], data['train_ce'], data['val_ce'], out_dir)
    summarize(folds, best_by_acc, best_by_loss, out_dir)

if __name__ == '__main__':
    main()
