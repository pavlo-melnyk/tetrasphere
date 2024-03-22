import sys
import os
from pathlib import Path

project_path = [p.parent for p in Path(__file__).resolve().parents if p.name == 'tetrasphere'][0]
sys.path.append(str(project_path))

from tetrasphere.experiments.cls_common import train_one_variant


def main(array_id):

    model_type = 'tetrasphere'
    num_spheres = 8

    epochs_per_hour = 15
    num_epochs = 250
    seed = 1

    dset_name, dset_split = "mn40", None

    i = 0
    for rot in ['z', 'so3']:

        full_name = Path(__file__).stem + f"--{model_type},K={num_spheres},rot={rot},epochs={num_epochs},seed={seed}"

        if i == array_id or array_id == -1:
            model_kwargs = dict(num_spheres=num_spheres)
            train_one_variant(__file__, full_name, num_epochs, seed, model_type, dset_name, rot,
                              dset_split=dset_split, model_kwargs=model_kwargs)

        if i == array_id:
            return

        i += 1

    return i, num_epochs, epochs_per_hour


if __name__ == '__main__':

    array_size, num_epochs, epochs_per_hour = main(-2)
    print(f"{array_size=}, {num_epochs=}, {epochs_per_hour=}")

    array_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', "-1"))
    main(array_id)
