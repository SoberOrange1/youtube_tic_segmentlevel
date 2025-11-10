import json
import os, re
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class TicNonTicSegmentDataset(Dataset):
    """
    Dataset for tic/non-tic classification from cropped face segments.

    Directory layout (root points to data_folder/cropped_faces or cropped_facces):
      root/
        tic/ | non-tic/
          <two-digit-id>/
            segment_xxx/
              <frames>.jpg
              [optional] blendshapes/ or landmarks/  # holds per-frame features (.npy/.txt/.csv)
              [optional] blendshapes.npy             # segment-level (T,D) features
              face_landmarks_blendshapes.json        # JSON file with all frame data
    """
    def __init__(self, root, target_len: int = 32, mode: str = 'train',
                 blend_dim: int = 52, exts=(".jpg", ".jpeg", ".png"),
                 use_blend: bool = True,
                 blend_folder_candidates=("blendshapes", "landmarks"),
                 blend_exts=(".npy", ".txt", ".csv"),
                 strict_blend: bool = False):
        self.root = root
        self.target_len = target_len
        self.mode = mode
        self.blend_dim = blend_dim
        self.exts = tuple([e.lower() for e in exts])
        self.use_blend = use_blend
        self.blend_folder_candidates = tuple(blend_folder_candidates)
        self.blend_exts = tuple(blend_exts)
        self.strict_blend = strict_blend

        # Transforms: remove Normalize to avoid double normalization.
        # ViTFeatureExtractor will handle resizing + its own normalization internally.
        if self.mode == 'train':
            self.frame_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),  # keep values in [0,1]
            ])
        else:
            self.frame_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

        self.class_to_idx = {"non-tic": 0, "tic": 1}
        self.samples = self._collect_samples()
        print(f" Collected {len(self.samples)} segment samples from {self.root}")

    def _is_two_digit_id(self, name: str) -> bool:
        return bool(re.match(r"^\d{2}$", name))

    def _collect_samples(self):
        samples = []
        for cls_name, cls_idx in self.class_to_idx.items():
            cls_dir = os.path.join(self.root, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for vid in sorted(os.listdir(cls_dir)):
                vid_dir = os.path.join(cls_dir, vid)
                if not os.path.isdir(vid_dir) or not self._is_two_digit_id(vid):
                    continue
                for seg in sorted(os.listdir(vid_dir)):
                    if not (seg.startswith("segment_")):
                        continue
                    seg_dir = os.path.join(vid_dir, seg)
                    if not os.path.isdir(seg_dir):
                        continue
                    has_frames = any(f.lower().endswith(self.exts) for f in os.listdir(seg_dir))
                    if has_frames:
                        samples.append((seg_dir, cls_idx))
        return samples

    def _find_blend_dir(self, seg_dir: str):
        for name in self.blend_folder_candidates:
            p = os.path.join(seg_dir, name)
            if os.path.isdir(p):
                return p
        return None

    def _load_blend_from_json(self, seg_dir: str, T: int):
        """
        Load blendshapes from a single JSON file in the segment directory.
        The JSON file is expected to be named 'face_landmarks_blendshapes.json'.
        """
        json_path = os.path.join(seg_dir, 'face_landmarks_blendshapes.json')
        if not os.path.isfile(json_path):
            return None

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except Exception:
            return None

        if 'frames' not in data or not isinstance(data['frames'], list):
            return None

        # Sort frames by frame_index to ensure correct temporal order
        frames_data = sorted(data['frames'], key=lambda x: x.get('frame_index', 0))
        
        if not frames_data:
            return None

        # Get blendshape keys from the first frame and sort them for consistency
        first_frame_blends = frames_data[0].get('face_blendshapes', {})
        if not first_frame_blends:
            return None
        
        blend_keys = sorted(first_frame_blends.keys())
        
        blend_vectors = []
        for frame_data in frames_data:
            blend_dict = frame_data.get('face_blendshapes', {})
            if blend_dict:
                # Extract values in consistent order, use 0.0 for missing keys
                vec = [blend_dict.get(key, 0.0) for key in blend_keys]
                blend_vectors.append(vec)
            else:
                # If a frame has no blendshapes, append zero vector
                blend_vectors.append([0.0] * len(blend_keys))

        if not blend_vectors:
            return None

        blends = np.array(blend_vectors, dtype=np.float32)  # shape (num_json_frames, D_original)

        # Resample to match target number of frames (T)
        if blends.shape[0] != T:
            if T > blends.shape[0]:
                # Need more frames - use interpolation indices
                interp_indices = np.linspace(0, blends.shape[0] - 1, T)
                blends_resampled = np.zeros((T, blends.shape[1]), dtype=np.float32)
                for i, idx in enumerate(interp_indices):
                    blends_resampled[i] = blends[int(np.round(idx))]
                blends = blends_resampled
            else:
                # Need fewer frames - subsample
                step = blends.shape[0] / T
                indices = [int(i * step) for i in range(T)]
                blends = blends[indices]

        # Pad or truncate feature dimension to match blend_dim
        D = self.blend_dim
        if blends.shape[1] < D:
            pad = np.zeros((blends.shape[0], D - blends.shape[1]), dtype=np.float32)
            blends = np.concatenate([blends, pad], axis=1)
        elif blends.shape[1] > D:
            blends = blends[:, :D]

        return blends

    def _load_blend_vector(self, file_path: str) -> np.ndarray:
        fp = file_path.lower()
        if fp.endswith('.npy'):
            arr = np.load(file_path)
        else:
            # txt/csv: assume whitespace or comma separated
            arr = np.loadtxt(file_path, delimiter=',' if fp.endswith('.csv') else None)
        arr = np.asarray(arr)
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        # pad/truncate to blend_dim
        D = self.blend_dim
        if arr.shape[0] < D:
            pad = np.zeros(D - arr.shape[0], dtype=arr.dtype)
            arr = np.concatenate([arr, pad], axis=0)
        elif arr.shape[0] > D:
            arr = arr[:D]
        return arr.astype(np.float32)

    def _try_load_segment_level_blend(self, seg_dir: str):
        # check segment root for a single file
        for name in ("blendshapes.npy", "landmarks.npy"):
            p = os.path.join(seg_dir, name)
            if os.path.isfile(p):
                try:
                    data = np.load(p)
                    return data  # expected shape (T,D) or (D,T)
                except Exception:
                    pass
        # also check inside blend folder
        bd = self._find_blend_dir(seg_dir)
        if bd is not None:
            for name in ("blendshapes.npy", "landmarks.npy"):
                p = os.path.join(bd, name)
                if os.path.isfile(p):
                    try:
                        data = np.load(p)
                        return data
                    except Exception:
                        pass
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seg_dir, label = self.samples[idx]
        # derive video id (two-digit folder) for per-video ranking grouping
        video_id = os.path.basename(os.path.dirname(seg_dir))  # two-digit string
        try:
            video_id_int = int(video_id)
        except ValueError:
            # fallback: hash if not numeric
            video_id_int = abs(hash(video_id)) % (10**6)

        # Frame Stream: load and sample frames from the segment directory
        frame_files = sorted([
            os.path.join(seg_dir, f)
            for f in os.listdir(seg_dir)
            if f.lower().endswith(self.exts)
        ])
        num_frames = len(frame_files)
        if num_frames == 0:
            raise ValueError(f"No valid frames in {seg_dir}")

        # sample to target_len frames to keep both streams in sync on time length
        target_len = int(self.target_len)
        if num_frames >= target_len:
            step = num_frames / target_len
            indices = [int(i * step) for i in range(target_len)]
        else:
            interp_indices = torch.linspace(0, num_frames - 1, steps=target_len)
            indices = interp_indices.round().long().clamp(0, num_frames - 1).tolist()

        frames = []
        for i in indices:
            with Image.open(frame_files[i]) as img:
                image = img.convert('RGB')
                frames.append(self.frame_transform(image))
        frame_tensor = torch.stack(frames, dim=0)  # [T, C, H, W]

        # Blend/landmarks stream
        T = len(indices)
        D = self.blend_dim
        blend = None
        if self.use_blend:
            # 1) Try loading from JSON file first (new format)
            blend = self._load_blend_from_json(seg_dir, T)
            
            if blend is None:
                # 2) segment-level file (T,D) or (D,T)
                seg_level = self._try_load_segment_level_blend(seg_dir)
                if seg_level is not None:
                    seg_arr = np.asarray(seg_level)
                    if seg_arr.ndim == 1:
                        # repeat single vector across time
                        v = self._load_blend_vector(seg_arr)
                        blend = np.tile(v[None, :], (T, 1))
                    elif seg_arr.ndim == 2:
                        # ensure shape (T,D)
                        if seg_arr.shape[0] == T or seg_arr.shape[1] == D:
                            if seg_arr.shape[1] != D:
                                # transpose or pad/truncate
                                seg_arr = seg_arr.T if seg_arr.shape[0] == D else seg_arr
                            # adjust D
                            if seg_arr.shape[1] != D:
                                if seg_arr.shape[1] > D:
                                    seg_arr = seg_arr[:, :D]
                                else:
                                    pad = np.zeros((seg_arr.shape[0], D - seg_arr.shape[1]), dtype=seg_arr.dtype)
                                    seg_arr = np.concatenate([seg_arr, pad], axis=1)
                            # sample by indices mapping from original frames length
                            # assume seg_arr length corresponds to available frames in folder
                            if seg_arr.shape[0] != T:
                                # resample by nearest
                                src_T = seg_arr.shape[0]
                                map_idx = np.clip(np.round(np.linspace(0, src_T - 1, num=T)).astype(int), 0, src_T - 1)
                                seg_arr = seg_arr[map_idx]
                            blend = seg_arr.astype(np.float32)
                        # else: ignore
                        
            if blend is None:
                # 3) per-frame files in a blend dir: match by stem, else by order
                bdir = self._find_blend_dir(seg_dir)
                if bdir is not None:
                    # map by stem
                    per_frame = []
                    all_blend_files = sorted([f for f in os.listdir(bdir) if f.lower().endswith(self.blend_exts)])
                    for i in indices:
                        stem = os.path.splitext(os.path.basename(frame_files[i]))[0]
                        cand = None
                        for ext in self.blend_exts:
                            p = os.path.join(bdir, stem + ext)
                            if os.path.isfile(p):
                                cand = p
                                break
                        if cand is None:
                            # fallback: align by order
                            if len(all_blend_files) > 0:
                                j = min(int(round(i * len(all_blend_files) / max(1, num_frames))), len(all_blend_files) - 1)
                                cand = os.path.join(bdir, all_blend_files[j])
                        if cand is not None:
                            try:
                                v = self._load_blend_vector(cand)
                                per_frame.append(v)
                            except Exception:
                                per_frame.append(np.zeros(D, dtype=np.float32))
                        else:
                            per_frame.append(np.zeros(D, dtype=np.float32))
                    blend = np.stack(per_frame, axis=0) if len(per_frame) else None
                    
        if blend is None:
            if self.strict_blend:
                raise FileNotFoundError(f"No blend/landmark features found for {seg_dir}")
            blend = np.zeros((T, D), dtype=np.float32)

        blend_tensor = torch.from_numpy(blend)  # [T, D]
        label = torch.tensor(label, dtype=torch.long)
        video_id_tensor = torch.tensor(video_id_int, dtype=torch.long)
        return frame_tensor, blend_tensor.transpose(0, 1), label, video_id_tensor  # [T,C,H,W], [D,T], label, video id
