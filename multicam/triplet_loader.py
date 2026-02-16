# multicam/triplet_loader.py
# Loader: yields synchronized nuScenes triplets (CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT)
# fetches the synchronized frames from the dataset and hand them in a consistent format 
import os
import cv2
from nuscenes.nuscenes import NuScenes


class TripletLoader:
    """
    Iterates through the first scene in nuScenes and yields:
      (t, left_img, front_img, right_img)
    Images are resized to `size` (w, h).
    """
    def __init__(self, dataroot: str, version: str = "v1.0-mini", size=(640, 360), scene_index: int = 0):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.size = size
        self.scene_index = scene_index

        if scene_index < 0 or scene_index >= len(self.nusc.scene):
            raise ValueError(f"scene_index out of range: {scene_index} (num scenes={len(self.nusc.scene)})")

        self.scene = self.nusc.scene[scene_index]
        self.token = self.scene["first_sample_token"]
        self.t = 0

    def _load_img(self, sample_data_record) -> "cv2.Mat":
        rel_path = sample_data_record["filename"]  # e.g. samples/CAM_FRONT/xxx.jpg
        path = os.path.join(self.nusc.dataroot, rel_path)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not read image at: {path}")
        return cv2.resize(img, self.size)

    def next(self):
        """Return next (t, left, front, right) or None when finished."""
        if not self.token:
            return None

        sample = self.nusc.get("sample", self.token)

        left_sd = self.nusc.get("sample_data", sample["data"]["CAM_FRONT_LEFT"])
        front_sd = self.nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        right_sd = self.nusc.get("sample_data", sample["data"]["CAM_FRONT_RIGHT"])

        left = self._load_img(left_sd)
        front = self._load_img(front_sd)
        right = self._load_img(right_sd)

        out = (self.t, left, front, right)

        self.token = sample["next"]
        self.t += 1
        return out
