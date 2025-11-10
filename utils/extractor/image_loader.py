from pathlib import Path
from typing import Iterable, List, Optional
import cv2
import numpy as np

DEFAULT_EXTS = (".jpg", ".jpeg", ".png")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_image_files(folder: Path, exts: Iterable[str] = DEFAULT_EXTS) -> List[Path]:
    exts = tuple([e.lower() for e in exts])
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def load_image_rgb(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def show_image(img_bgr: np.ndarray, title: str = "Image", save_path: Optional[str] = None, show: bool = False):
    if save_path:
        cv2.imwrite(save_path, img_bgr)
    if show:
        cv2.imshow(title, img_bgr)
        cv2.waitKey(1)
