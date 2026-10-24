"""Download a few lidar frames from the free KITTI raw dataset.

The full zip for drive 2011_09_26_0001 is ~460 MB (cameras + lidar + GPS).
This script uses HTTP range requests to pull ONLY the files we need
(the first N Velodyne scans + GPS/IMU "oxts" records), about 2 MB per frame.

KITTI is released under CC BY-NC-SA 3.0 (non-commercial). Please cite:
  Geiger et al., "Vision meets Robotics: The KITTI Dataset", IJRR 2013.

Run:  python tools/download_kitti.py            (first 30 frames)
      python tools/download_kitti.py --frames 108   (whole drive)
"""
import argparse
import io
import os
import urllib.request
import zipfile

DRIVE = "2011_09_26_drive_0001"
URL = f"https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/{DRIVE}/{DRIVE}_sync.zip"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "kitti", DRIVE)


class HttpRangeFile(io.RawIOBase):
    """A read-only, seekable file object backed by HTTP range requests."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, buf):
        n = min(len(buf), self.size - self.pos)
        if n <= 0:
            return 0
        rng = f"bytes={self.pos}-{self.pos + n - 1}"
        req = urllib.request.Request(self.url, headers={"Range": rng})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        buf[:len(data)] = data
        self.pos += len(data)
        return len(data)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=30)
    args = ap.parse_args()

    # Buffer reads so zipfile's many small reads don't become many HTTP requests.
    remote = io.BufferedReader(HttpRangeFile(URL), buffer_size=1 << 20)
    zf = zipfile.ZipFile(remote)
    wanted = []
    for name in zf.namelist():
        stem = os.path.splitext(os.path.basename(name))[0]
        if "/velodyne_points/" in name or "/oxts/" in name:
            if not stem.isdigit() or int(stem) < args.frames:
                wanted.append(name)

    for i, name in enumerate(sorted(wanted)):
        rel = name.split(f"{DRIVE}_sync/", 1)[-1]
        dst = os.path.join(OUT, rel)
        if name.endswith("/") or os.path.exists(dst):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with zf.open(name) as src, open(dst, "wb") as f:
            f.write(src.read())
        print(f"[{i + 1}/{len(wanted)}] {rel}")
    print("done ->", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
