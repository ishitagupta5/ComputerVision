# multicam/stereo_loader.py
import os
import cv2

IMG_EXTS = (".png", ".jpg", ".jpeg")

def list_images(root):
    """List images in root recursively, sorted."""
    out = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(IMG_EXTS):
                out.append(os.path.join(dirpath, f))
    out.sort()
    return out

class StereoKittiLoader:
    """
    Stereo loader expecting:
      dataroot/image_2/... (left)
      dataroot/image_3/... (right)

    Pairing strategy:
      1) If left/right have matching relative paths, use that.
      2) Otherwise fall back to pairing by sorted index (best for hand-picked sets).

    Returns: (t, left_bgr, right_bgr)
    """
    def __init__(self, dataroot, size=(640, 360)):
        self.dataroot = dataroot
        self.size = size

        self.left_dir = os.path.join(dataroot, "image_2")
        self.right_dir = os.path.join(dataroot, "image_3")

        if not os.path.isdir(self.left_dir) or not os.path.isdir(self.right_dir):
            raise FileNotFoundError(
                f"Expected folders:\n  {self.left_dir}\n  {self.right_dir}"
            )

        self.left_files = list_images(self.left_dir)
        self.right_files = list_images(self.right_dir)

        if not self.left_files:
            raise FileNotFoundError(f"No images found in {self.left_dir}")
        if not self.right_files:
            raise FileNotFoundError(f"No images found in {self.right_dir}")

        # Build a right lookup by relative path (for datasets that match structure)
        self.right_rel_map = {}
        for r in self.right_files:
            rel = os.path.relpath(r, self.right_dir)
            self.right_rel_map[rel] = r

        self.idx = 0

    def restart(self):
        self.idx = 0

    def next(self):
        # Iterate left frames in order; pair with matching right if possible, else by index
        while self.idx < len(self.left_files):
            lpath = self.left_files[self.idx]
            rel = os.path.relpath(lpath, self.left_dir)

            # First try: same relative path
            rpath = self.right_rel_map.get(rel, None)

            # Fallback: pair by index (works for manual 10-image sets)
            if rpath is None:
                if self.idx >= len(self.right_files):
                    return None
                rpath = self.right_files[self.idx]

            self.idx += 1

            left = cv2.imread(lpath, cv2.IMREAD_COLOR)
            right = cv2.imread(rpath, cv2.IMREAD_COLOR)
            if left is None or right is None:
                continue

            if self.size is not None:
                left = cv2.resize(left, self.size, interpolation=cv2.INTER_AREA)
                right = cv2.resize(right, self.size, interpolation=cv2.INTER_AREA)

            t = self.idx - 1
            return (t, left, right)

        return None