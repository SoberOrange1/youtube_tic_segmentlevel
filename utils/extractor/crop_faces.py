import cv2
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm


def crop_faces_from_data_folder(data_root, dst_root, dataset='tic', video_ids=None,
                                image_size=224, min_detection_confidence=0.5,
                                save_original_when_no_face=True, exts=('.jpg', '.jpeg', '.png')):
    """
    Traverse data_root/<dataset>/<video_id>/segment_<nnn>/ and run MediaPipe face detection
    on every frame image. Save cropped faces (resized to image_size) to:
      dst_root/<dataset>/<video_id>/segment_<nnn>/<same-filename>

    Call this function directly (no argparse). Typical usage: edit variables in __main__ and run the file.
    """
    data_root = Path(data_root)
    dst_root = Path(dst_root)
    src_dataset_dir = data_root / dataset
    if not src_dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {src_dataset_dir}")

    video_dirs = sorted([p for p in src_dataset_dir.iterdir() if p.is_dir()])
    if video_ids:
        # Only accept two-digit numeric or two-character ids (e.g. '02', '14').
        # Ignore single-digit inputs to avoid accidental matches.
        norm_ids = set()
        for v in video_ids:
            s = str(v)
            if s.isdigit() and len(s) == 2:
                norm_ids.add(s)
            elif s.isdigit() and len(s) == 1:
                print(f"Warning: ignoring single-digit video id '{s}' — expected two-digit id like '02'")
            else:
                if len(s) == 2:
                    norm_ids.add(s)
                else:
                    print(f"Warning: ignoring video id '{s}' — expected two-digit string")

        video_dirs = [p for p in video_dirs if p.name in norm_ids]
        if not video_dirs:
            print(f"Warning: no matching two-digit video folders found for requested ids: {video_ids}")

    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=min_detection_confidence) as face_detection:
        for video_dir in tqdm(video_dirs, desc="Videos"):
            for segment_dir in sorted([d for d in video_dir.iterdir() if d.is_dir() and d.name.startswith("segment_")]):
                out_segment_dir = dst_root / dataset / video_dir.name / segment_dir.name
                out_segment_dir.mkdir(parents=True, exist_ok=True)

                frame_files = sorted([f for f in segment_dir.iterdir() if f.suffix.lower() in exts])
                for frame_path in frame_files:
                    img = cv2.imread(str(frame_path))
                    if img is None:
                        print(f"Warning: cannot read {frame_path}")
                        continue
                    h, w = img.shape[:2]
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = face_detection.process(rgb)

                    out_path = out_segment_dir / frame_path.name

                    if results.detections and len(results.detections) > 0:
                        # use the largest detection by area
                        best = None
                        best_area = 0
                        for det in results.detections:
                            r = det.location_data.relative_bounding_box
                            area = max(0, r.width * r.height)
                            if area > best_area:
                                best_area = area
                                best = r

                        det = best
                        xmin = max(0, int(det.xmin * w))
                        ymin = max(0, int(det.ymin * h))
                        box_w = int(det.width * w)
                        box_h = int(det.height * h)

                        pad_w = int(0.15 * box_w)
                        pad_h = int(0.20 * box_h)
                        x0 = max(0, xmin - pad_w)
                        y0 = max(0, ymin - pad_h)
                        x1 = min(w, xmin + box_w + pad_w)
                        y1 = min(h, ymin + box_h + pad_h)

                        if x1 <= x0 or y1 <= y0:
                            crop = img
                        else:
                            crop = img[y0:y1, x0:x1]

                        try:
                            resized = cv2.resize(crop, (image_size, image_size))
                        except Exception:
                            resized = cv2.resize(img, (image_size, image_size))
                        cv2.imwrite(str(out_path), resized)
                    else:
                        if save_original_when_no_face:
                            resized = cv2.resize(img, (image_size, image_size))
                            cv2.imwrite(str(out_path), resized)
                        else:
                            # skip saving
                            continue


if __name__ == "__main__":
    # Edit these variables as needed, then run the script.
    DATA_ROOT = Path(r"a:\youtube_fine_tune\data_folder")  # contains tic/ and non-tic/
    DST_ROOT = Path(r"a:\youtube_fine_tune\data_folder\cropped_faces")
    DATASET = 'non-tic'  # 'tic' or 'non-tic'
    # Provide two-digit ids only (strings like '02', '03'). Single-digit numbers will be ignored.
    VIDEO_IDS = ['02','03','04','12','14','15']  # or None for all videos
    IMAGE_SIZE = 224

    crop_faces_from_data_folder(
        data_root=DATA_ROOT,
        dst_root=DST_ROOT,
        dataset=DATASET,
        video_ids=VIDEO_IDS,
        image_size=IMAGE_SIZE,
        min_detection_confidence=0.5,
        save_original_when_no_face=False
    )
