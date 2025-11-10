import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTModel, ViTConfig
from pytorch_tcn import TCN
from PIL import Image

# added for LoRA
from peft import get_peft_model, LoraConfig, TaskType

# add aggregation utility for new experimental model
from utils.segment_aggregation import score_segment

# ---------------- ViT Encoder (with optional LoRA) ---------------- #
class ViTFrameEncoder(nn.Module):
    def __init__(self, model_name='trpakov/vit-face-expression', device=None, dropout=0.4,
                 use_lora: bool = False, lora_r: int = 8, lora_alpha: int = 32,
                 lora_dropout: float = 0.1, lora_target_modules: list = None):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = ViTConfig.from_pretrained(model_name)
        self.config.hidden_dropout_prob = dropout
        self.model = ViTModel.from_pretrained(model_name, config=self.config)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.use_lora = use_lora
        if use_lora:
            target_modules = lora_target_modules or ["query", "value"]
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION
            )
            self.model = get_peft_model(self.model, lora_config)
        # Enable gradient checkpointing to save memory (works for HF ViT and PEFT-wrapped models)
        try:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'gradient_checkpointing_enable'):
                self.model.model.gradient_checkpointing_enable()
            # Also set flag in config if available
            if hasattr(self.model, 'config') and self.model.config is not None:
                setattr(self.model.config, 'gradient_checkpointing', True)
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'config') and self.model.model.config is not None:
                setattr(self.model.model.config, 'gradient_checkpointing', True)
        except Exception:
            pass
        self.model.to(self.device)
        self.hidden_dim = self.config.hidden_size

    def forward(self, images):
        """Encode a list (or batch) of PIL Images -> (N, hidden_dim)."""
        # Accept either PIL.Image / list[PIL.Image] or a precomputed tensor (pixel_values)
        device = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
        if isinstance(images, Image.Image):
            images = [images]
        if isinstance(images, torch.Tensor):
            # assume already pixel_values and move to device
            pixel_values = images.to(device)
        else:
            inputs = self.feature_extractor(images=images, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(device)
        with torch.set_grad_enabled(self.training and self.use_lora):
            # Explicitly pass pixel_values. When using PEFT, call the underlying HF model
            # to avoid wrappers introducing text-only kwargs like input_ids.
            if self.use_lora and hasattr(self.model, 'model'):
                outputs = self.model.model(pixel_values=pixel_values)
            else:
                outputs = self.model(pixel_values=pixel_values)
            # prefer last_hidden_state CLS token; fallback to pooler_output or outputs[0]
            if hasattr(outputs, 'last_hidden_state'):
                cls_embedding = outputs.last_hidden_state[:, 0, :]
            elif isinstance(outputs, dict) and 'last_hidden_state' in outputs:
                cls_embedding = outputs['last_hidden_state'][:, 0, :]
            elif hasattr(outputs, 'pooler_output'):
                cls_embedding = outputs.pooler_output
            else:
                cls_embedding = outputs[0][:, 0, :]
        return cls_embedding  # (N, hidden_dim)

# ---------------- Temporal TCN ---------------- #
class TemporalTCN(nn.Module):
    def __init__(self, tcn_config: dict):
        super().__init__()
        # Extract pooling_type before passing the rest to TCN so config is respected
        pooling = tcn_config.pop('pooling_type', 'none')
        # Build TCN with remaining config
        self.tcn = TCN(**tcn_config)
        self.output_dim = tcn_config.get('output_projection', tcn_config['num_channels'][-1])
        # pooling_type: 'none' (no internal temporal pooling, return pooled=None), 'avg', 'max', 'interval_avg'
        self.pooling_type = pooling

    def forward(self, x, interval: tuple = None, return_sequence: bool = False):
        """
        Forward pass with optional temporal pooling. When pooling_type == 'none' we do not pool
        and return pooled=None (caller should handle segment-level aggregation).
        :param x: Input tensor of shape [B, D, T] (Batch, Features, Time).
        :param interval: Optional tuple specifying the start and end of the interval.
        :param return_sequence: If True, also return the temporal feature map before pooling.
        :return: pooled (and optionally sequence) features. If pooling_type == 'none' pooled is None.
        """
        x = self.tcn(x)
        seq = x
        if interval is not None:
            start, end = interval
            x = x[:, :, start:end]
        if self.pooling_type == 'none':
            x_pooled = None
        elif self.pooling_type == 'avg':
            x_pooled = x.mean(dim=-1)
        elif self.pooling_type == 'max':
            x_pooled, _ = x.max(dim=-1)
        elif self.pooling_type == 'interval_avg':
            x_pooled = x.mean(dim=-1) if interval is None else x[:, :, start:end].mean(dim=-1)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")
        if return_sequence:
            return seq, x_pooled
        return x_pooled

# ---------------- Late Fusion Model ---------------- #
class LateFusionModel(nn.Module):
    def __init__(
        self,
        vit_encoder: ViTFrameEncoder,
        vit_tcn_config: dict,
        blend_tcn_config: dict,
        num_classes: int,
        fusion_mode: str = 'concat'
    ):
        super().__init__()
        self.vit_encoder = vit_encoder  # now an nn.Module so parameters register
        self.vit_tcn = TemporalTCN(vit_tcn_config)
        self.blend_tcn = TemporalTCN(blend_tcn_config)
        self.fusion_mode = fusion_mode
        if fusion_mode == 'concat':
            fusion_dim = self.vit_tcn.output_dim + self.blend_tcn.output_dim
        elif fusion_mode == 'add':
            fusion_dim = self.vit_tcn.output_dim
        else:
            raise ValueError(f"Unsupported fusion mode: {self.fusion_mode}")
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        self.per_frame_classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, frame_tensor_sequence, blendshape_sequence, return_frame_logits: bool = False):
        B, T, C, H, W = frame_tensor_sequence.shape
        # Convert all frames to PIL and encode in one batch for efficiency
        flat_frames = frame_tensor_sequence.view(-1, C, H, W)
        pil_images = [transforms.ToPILImage()(frm.cpu()) for frm in flat_frames]
        cls_embeddings = self.vit_encoder(pil_images)  # (B*T, D_vit)
        vit_features = cls_embeddings.view(B, T, -1).transpose(1, 2)  # [B, D_vit, T]
        # Get sequence outputs from both streams. The TemporalTCN may return pooled=None
        vit_seq, vit_pooled = self.vit_tcn(vit_features, return_sequence=True)
        blend_seq, blend_pooled = self.blend_tcn(blendshape_sequence, return_sequence=True)
        # Fuse sequences (per-frame features)
        if self.fusion_mode == 'concat':
            fused_seq = torch.cat([vit_seq, blend_seq], dim=1)  # [B, D_fusion, T]
        else:  # add
            fused_seq = vit_seq + blend_seq

        # Per-frame logits
        frame_logits = self.per_frame_classifier(fused_seq.permute(0, 2, 1))  # [B, T, num_classes]

        # Segment logits: compute by averaging per-frame logits in logit-space
        segment_logits = frame_logits.mean(dim=1)  # [B, num_classes]

        # If caller requests per-frame logits, return only per-frame logits and per-frame tic probabilities
        if return_frame_logits:
            per_frame_probs = F.softmax(frame_logits, dim=-1)[..., 1]  # [B, T]
            return frame_logits, per_frame_probs
        # Otherwise return segment-level logits (computed from per-frame logits)
        return segment_logits

# ---------------- Frame Prob Avg Fusion Model ---------------- #
class FrameProbAvgFusionModel(nn.Module):
    """
    Experimental model variant (no early temporal pooling) producing per-frame tic probabilities per stream.
    For each frame t:
      - Extract ViT CLS embedding -> pass through ViT TCN -> per-frame feature (vit_seq[:,:,t])
      - Pass blendshape sequence through its TCN -> per-frame feature (blend_seq[:,:,t])
      - Each stream gets its own per-frame classifier -> logits_vit_t, logits_blend_t
      - Convert to probabilities (softmax) and average across the two streams per frame:
            probs_avg_t = (probs_vit_t + probs_blend_t) / 2
    Returns per-frame logits and averaged probabilities.
    Keeps original LateFusionModel intact for comparison.
    """
    def __init__(self,
                 vit_encoder: ViTFrameEncoder,
                 vit_tcn_config: dict,
                 blend_tcn_config: dict,
                 num_classes: int = 2):
        super().__init__()
        assert num_classes == 2, "Current implementation assumes binary classification (tic / non-tic)."
        self.vit_encoder = vit_encoder
        self.vit_tcn = TemporalTCN(vit_tcn_config)
        self.blend_tcn = TemporalTCN(blend_tcn_config)
        self.num_classes = num_classes
        # Per-frame classifiers per stream (no late fusion pooling)
        self.vit_frame_classifier = nn.Linear(self.vit_tcn.output_dim, num_classes)
        self.blend_frame_classifier = nn.Linear(self.blend_tcn.output_dim, num_classes)

    def forward(self, frame_tensor_sequence, blendshape_sequence):
        """
        Args:
          frame_tensor_sequence: [B,T,C,H,W]
          blendshape_sequence:   [B,Db,T]
        Returns:
          dict with keys:
            per_frame_logits_vit:   [B,T,2]
            per_frame_logits_blend: [B,T,2]
            per_frame_probs_avg:    [B,T] (tic probability after averaging streams)
        """
        B, T, C, H, W = frame_tensor_sequence.shape
        # ---- ViT stream (batched frame encoding) ----
        flat_frames = frame_tensor_sequence.view(-1, C, H, W)
        pil_images = [transforms.ToPILImage()(frm.cpu()) for frm in flat_frames]
        cls_embeddings = self.vit_encoder(pil_images)  # (B*T, D_vit)
        vit_features = cls_embeddings.view(B, T, -1).transpose(1, 2)  # [B, D_vit, T]
        vit_seq, _ = self.vit_tcn(vit_features, return_sequence=True)  # [B, D_vit_out, T]
        vit_seq_t = vit_seq.permute(0, 2, 1)  # [B,T,D_vit_out]
        per_frame_logits_vit = self.vit_frame_classifier(vit_seq_t)  # [B,T,2]
        # ---- Blend stream ----
        blend_seq, _ = self.blend_tcn(blendshape_sequence, return_sequence=True)  # [B, D_blend_out, T]
        blend_seq_t = blend_seq.permute(0, 2, 1)  # [B,T,D_blend_out]
        per_frame_logits_blend = self.blend_frame_classifier(blend_seq_t)  # [B,T,2]
        # ---- Average probabilities per frame ----
        probs_vit = F.softmax(per_frame_logits_vit, dim=-1)
        probs_blend = F.softmax(per_frame_logits_blend, dim=-1)
        probs_avg = (probs_vit + probs_blend) / 2.0  # [B,T,2]
        tic_prob_avg = probs_avg[..., 1]  # [B,T]
        out = {
            'per_frame_logits_vit': per_frame_logits_vit,
            'per_frame_logits_blend': per_frame_logits_blend,
            'per_frame_probs_avg': tic_prob_avg,
        }
        return out
