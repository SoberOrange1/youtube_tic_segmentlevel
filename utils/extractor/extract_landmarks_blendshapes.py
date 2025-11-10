from __future__ import annotations
from pathlib import Path
import json
import re
from typing import Dict, List, Optional
from tqdm import tqdm
import numpy as np
import cv2

import mediapipe as mp

# Optional visualization imports; gracefully degrade if unavailable
try:
    from visualization_utils import draw_landmarks_on_image, plot_face_blendshapes_bar_graph
except Exception:
    def draw_landmarks_on_image(image, detection_result):  # type: ignore
        return image
    def plot_face_blendshapes_bar_graph(*args, **kwargs):  # type: ignore
        return None


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_image_files(dir_path: Path, exts=(".jpg", ".jpeg", ".png")) -> List[Path]:
    files = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in exts]
    # Simple natural-like sort by extracting first integer in filename if present
    def sort_key(p: Path):
        m = re.search(r"(\d+)", p.stem)
        return (int(m.group(1)) if m else float('inf'), p.name)
    return sorted(files, key=sort_key)


def load_image_rgb(path: Path):
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def is_two_digit_id(name: str) -> bool:
    return bool(re.match(r"^\d{2}$", name))


def find_segment_dirs(root: Path, subset: str) -> List[Path]:
    """
    root layout: <root>/{tic,non-tic}/{two-digit-id}/segment_xxx/
    subset in {'tic','non-tic','all'}
    """
    subsets = ['tic', 'non-tic'] if subset == 'all' else [subset]
    seg_dirs: List[Path] = []
    for sub in subsets:
        sub_dir = root / sub
        if not sub_dir.is_dir():
            continue
        for vid_dir in sorted([p for p in sub_dir.iterdir() if p.is_dir() and is_two_digit_id(p.name)]):
            for seg_dir in sorted([p for p in vid_dir.iterdir() if p.is_dir() and p.name.startswith("segment_")]):
                seg_dirs.append(seg_dir)
    return seg_dirs


def serialize_landmarks(face_landmarks) -> List[List[Dict[str, float]]]:
    """
    Returns a list for all faces; each is a list of dicts {x,y,z}
    """
    all_faces = []
    for lmks in face_landmarks:
        lst = []
        for lm in lmks:
            lst.append({"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)})
        all_faces.append(lst)
    return all_faces


def serialize_blendshapes(face_blendshapes) -> Optional[Dict[str, float]]:
    """
    Use the first face's blendshapes (if present), map {name: score}
    """
    if not face_blendshapes or len(face_blendshapes) == 0:
        return None
    m = {}
    for cat in face_blendshapes[0]:
        m[cat.category_name] = float(cat.score)
    return m


def serialize_transformation_matrixes(facial_matrixes) -> Optional[List[List[List[float]]]]:
    """
    Serialize per-face facial transformation matrices, if present.
    Tries to handle MediaPipe's Matrix container (rows, cols, data) or nested lists.
    Returns list of 2D lists per face.
    """
    if not facial_matrixes:
        return None
    out = []
    for m in facial_matrixes:
        try:
            # MediaPipe Matrix container
            data = list(getattr(m, 'data'))
            rows = int(getattr(m, 'rows', 4))
            cols = int(getattr(m, 'cols', 4))
            mat = [[float(data[r * cols + c]) for c in range(cols)] for r in range(rows)]
            out.append(mat)
        except Exception:
            try:
                # Fallback for array-like
                mat = np.array(m, dtype=float).tolist()
                out.append(mat)
            except Exception:
                out.append(None)
    return out if out else None


def process_segment(
    seg_dir: Path,
    detector,
    visualize: bool = False,
    exts=(".jpg", ".jpeg", ".png"),
    frame_rate: int = 30,
) -> Dict:
    """
    Process all frames in a segment dir; return JSON dict.
    Save visualizations (if enabled) under seg_dir/vis/.
    """
    frame_files = list_image_files(seg_dir, exts)
    if len(frame_files) == 0:
        return {"segment": seg_dir.name, "frames": []}

    vis_dir = ensure_dir(seg_dir / "vis") if visualize else None

    frames_json = []
    for idx, frame_path in enumerate(frame_files):
        rgb = load_image_rgb(frame_path)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        timestamp_ms = int(round(idx * 1000.0 / max(frame_rate, 1)))
        blend = serialize_blendshapes(result.face_blendshapes)

        frame_entry = {
            "file": frame_path.name,
            "frame_index": idx,
            "timestamp_ms": timestamp_ms,
            "face_count": len(result.face_landmarks),
            "face_landmarks": serialize_landmarks(result.face_landmarks),
            "face_blendshapes": (blend if blend is not None else []),
            "facial_transformation_matrixes": serialize_transformation_matrixes(getattr(result, 'facial_transformation_matrixes', None)),
        }
        frames_json.append(frame_entry)

        if visualize:
            annotated = draw_landmarks_on_image(rgb, result)
            bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str((vis_dir / f"annotated_{idx:03d}.png")), bgr)

            if result.face_blendshapes:
                plot_face_blendshapes_bar_graph(
                    result.face_blendshapes[0],
                    save_path=str(vis_dir / f"blendshapes_{idx:03d}.png"),
                )

    return {
        "segment": seg_dir.name,
        "frame_count": len(frame_files),
        "frames": frames_json,
    }


def run_extraction(
    root: str,
    subset: str = "all",
    model_path: str = "./face_landmarker.task",
    visualize: bool = False,
    exts: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    limit: int = 0,
    overwrite: bool = False,
    num_faces: int = 1,
    output_face_blendshapes: bool = False,
    output_facial_transformation_matrixes: bool = False,
    frame_rate: int = 30,
):
    """
    Main logic for extracting landmarks and blendshapes.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Root not found: {root_path}")

    # Create detector once
    base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        output_face_blendshapes=output_face_blendshapes,
        output_facial_transformation_matrixes=output_facial_transformation_matrixes,
        num_faces=num_faces,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
    )
    detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    # Determine segments: allow root to be a direct frames folder, otherwise scan dataset layout
    has_direct_frames = False
    try:
        has_direct_frames = len(list_image_files(root_path, exts=exts)) > 0
    except Exception:
        has_direct_frames = False

    if has_direct_frames:
        seg_dirs = [root_path]
    else:
        seg_dirs = find_segment_dirs(root_path, subset)

    if limit > 0:
        seg_dirs = seg_dirs[:limit]

    pbar = tqdm(seg_dirs, desc="Segments")
    current_video = ""
    for seg_dir in pbar:
        # Update progress bar to show current video (classification/id)
        try:
            video_id = seg_dir.parent.name
            classification = seg_dir.parent.parent.name
            if is_two_digit_id(video_id):
                video_info = f"{classification}/{video_id}"
                if video_info != current_video:
                    current_video = video_info
                    pbar.set_description(f"Processing {current_video}")
        except IndexError:
            # Could happen on a very shallow path, ignore.
            pass

        try:
            out_path = seg_dir / "face_landmarks_blendshapes.json"
            if out_path.exists() and not overwrite:
                continue
            json_dict = process_segment(
                seg_dir,
                detector,
                visualize=visualize,
                exts=exts,
                frame_rate=frame_rate,
            )
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(json_dict, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[WARN] Failed on {seg_dir}: {e}")

    print("Done.")


def main():
    # Parameters are now defined directly in the code.
    # Modify these values to change the script's behavior.
    run_extraction(
        root="a:/youtube_fine_tune/data_folder/cropped_faces",  # Root folder or a direct folder with frames.
        subset="all",  # "tic", "non-tic", or "all"
        model_path="./face_landmarker.task",
        visualize=True,
        exts=(".jpg", ".jpeg", ".png"),
        limit=0,  # 0 for no limit
        overwrite=False,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        frame_rate=30,
    )


if __name__ == "__main__":
    main()
