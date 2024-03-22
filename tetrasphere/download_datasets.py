# Convenience tool for downloading the datasets.
# Requires ~30 GB disk space uncompressed, including the zip-files.

import os
from pathlib import Path
import shutil
import sys

project_path = [p.parent for p in Path(__file__).resolve().parents if p.name == 'tetrasphere'][0]
sys.path.append(str(project_path))

from tetrasphere.config import Environment

if __name__ == "__main__":

    dset_path = Environment.dset_path
    dset_path.mkdir(exist_ok=True)
    os.chdir(dset_path)

    def download(url):
        filename = dset_path / url.split("/")[-1]
        if not filename.exists():
            os.system(f"wget --no-check-certificate {url}")
        return filename

    mn40 = download("https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip")
    sonn = download("http://hkust-vgd.ust.hk/scanobjectnn/h5_files.zip")
    snps = download("https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip")

    os.system(f"unzip -o {mn40}")

    os.system(f"unzip -o {sonn}")
    shutil.move("h5_files", "scanobjectnn")

    os.system(f"unzip -o {snps}")
    shutil.move("hdf5_data", "shapenet_part_seg")
