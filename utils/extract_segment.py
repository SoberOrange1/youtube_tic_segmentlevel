import os
import re
from pathlib import Path
import cv2


def extract_segments_from_folder(folder_path, target_fps=1, segment_seconds=1, dst_root=None, verbose=True):
    """
    Read all .mp4 files from folder_path and split each video into segments of segment_seconds seconds.
    Each segment's frames are saved under dst_root/<video_id>/segment_<seg_id>/ (if dst_root is None, folder_path is used).
    Only full segments are processed; any leftover segment shorter than segment_seconds is discarded.
    The video_id is taken from the first two digits of the filename (if present), otherwise the first two characters.

    Parameters:
      folder_path: str or Path, folder containing mp4 files
      target_fps: int, number of frames to save per second within each segment
      segment_seconds: int or float, length of each segment in seconds (the user requested 1 second)
      dst_root: optional, root output path (defaults to folder_path)
      verbose: bool, whether to print progress information
    """
    folder = Path(folder_path)
    if dst_root is None:
        dst_root = folder
    else:
        dst_root = Path(dst_root)

    mp4_files = sorted(folder.glob("*.mp4"))
    if not mp4_files:
        if verbose:
            print("No mp4 files found in", folder)
        return

    # Track next segment index per video_id so segments remain continuous across multiple files
    video_segment_counters = {}

    for mp4 in mp4_files:
        fname = mp4.stem  # filename without extension
        m = re.match(r"\d{2}", fname)
        video_id = m.group(0) if m else fname[:2]

        # starting index for segments of this video_id
        start_seg_idx = video_segment_counters.get(video_id, 0)

        cap = cv2.VideoCapture(str(mp4))
        if not cap.isOpened():
            if verbose:
                print(f"Failed to open {mp4}")
            continue

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if video_fps <= 0 or total_frames <= 0:
            if verbose:
                print(f"Cannot determine fps/frames for {mp4}, skipping.")
            cap.release()
            continue

        duration = total_frames / video_fps
        full_segments = int(duration // segment_seconds)
        if verbose:
            print(f"Processing {mp4.name}: fps={video_fps:.2f}, frames={total_frames}, duration={duration:.2f}s, full_segments={full_segments}")

        for seg_idx in range(full_segments):
            # compute a global segment index so numbering continues across files with the same video_id
            global_seg_idx = start_seg_idx + seg_idx
            seg_dir = dst_root / video_id / f"segment_{global_seg_idx:03d}"
            seg_dir.mkdir(parents=True, exist_ok=True)

            # Sample target_fps frames uniformly within the segment duration
            saved = 0
            seen_frame_idxs = set()
            for k in range(int(target_fps)):
                # time offset relative to the start of the segment
                t = (seg_idx * segment_seconds) + (k / target_fps)
                frame_idx = int(round(t * video_fps))
                # ensure frame index stays within this segment range
                seg_start_frame = int(round(seg_idx * segment_seconds * video_fps))
                seg_end_frame = int(round((seg_idx + 1) * segment_seconds * video_fps)) - 1
                if frame_idx < seg_start_frame:
                    frame_idx = seg_start_frame
                if frame_idx > seg_end_frame:
                    frame_idx = seg_end_frame
                if frame_idx in seen_frame_idxs:
                    continue
                seen_frame_idxs.add(frame_idx)

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                out_name = seg_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(out_name), frame)
                saved += 1

            if verbose:
                print(f"  video {video_id} segment {global_seg_idx:03d}: saved {saved} frames -> {seg_dir}")

        # update the next segment index for this video_id so subsequent files continue numbering
        video_segment_counters[video_id] = start_seg_idx + full_segments

        cap.release()



if __name__ == "__main__":
    # Example usage: modify the root path to the folder that contains your mp4 files
    root = Path(r"E:\HCI\summer_intern_KIXLAB\youtube_videos\clipped videos\tic\08")  # change to the directory containing mp4 files
    target = Path(r"A:\youtube_fine_tune\data_folder\tic")  # change to your desired output directory
    fps = 30  # target frames per second
    extract_segments_from_folder(folder_path=root, target_fps=fps, segment_seconds=1, dst_root=target, verbose=True)
