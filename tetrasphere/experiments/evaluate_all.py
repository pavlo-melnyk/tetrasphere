from pathlib import Path
import sys

project_path = [p.parent for p in Path(__file__).resolve().parents if p.name == 'tetrasphere'][0]
sys.path.append(str(project_path))

from tetrasphere.config import Environment
from tetrasphere.experiments.cls_common import train_one_variant as eval_cls
from tetrasphere.experiments.partseg_so3 import eval_main as eval_partseg

def evaluate_mn40():
    model_type = 'tetrasphere'
    num_spheres = 8

    rot = "z"
    num_epochs = 250
    seed = 3

    dset_name, dset_split = "mn40", None

    identifier = f"{model_type}_K={num_spheres}_{rot}aug_{dset_name}"
    model_path = Environment.weights_path / f"cvpr2024_{identifier}_90_5.ckpt"
    full_name = f"eval_{identifier}"

    print(f"Evaluating {model_path.stem}")
    print("-------------------------------------")
    eval_cls(__file__, full_name, num_epochs, seed, model_type, dset_name, rot=rot, batch_size=32,
             dset_split=dset_split, model_kwargs=dict(num_spheres=num_spheres), run_test=True, test_ckpt=model_path)


def evaluate_objbg():
    model_type = 'tetrasphere'
    num_spheres = 2

    rot = "z"
    num_epochs = 250
    seed = 3

    dset_name, dset_split = "sobjnn", ("main_split", "obj")

    identifier = f"{model_type}_K={num_spheres}_{rot}aug_{dset_name}_objbg"
    model_path = Environment.weights_path / f"cvpr2024_{identifier}_87_3.ckpt"
    full_name = f"eval_{identifier}"

    print(f"Evaluating {model_path.stem}")
    print("-------------------------------------")
    eval_cls(__file__, full_name, num_epochs, seed, model_type, dset_name, rot=rot, batch_size=32,
             dset_split=dset_split, model_kwargs=dict(num_spheres=num_spheres), run_test=True, test_ckpt=model_path)


def evaluate_pbt50rs():
    model_type = 'tetrasphere'
    num_spheres = 4

    rot = "z"
    num_epochs = 250
    seed = 3

    dset_name, dset_split = "sobjnn", ("main_split", "pb_t50_rs")

    identifier = f"{model_type}_K={num_spheres}_{rot}aug_{dset_name}_pbt50rs"
    model_path = Environment.weights_path / f"cvpr2024_{identifier}_79_2.ckpt"
    full_name = f"eval_{identifier}"

    print(f"Evaluating {model_path.stem}")
    print("-------------------------------------")
    eval_cls(__file__, full_name, num_epochs, seed, model_type, dset_name, rot=rot, batch_size=32,
             dset_split=dset_split, model_kwargs=dict(num_spheres=num_spheres), run_test=True, test_ckpt=model_path)


def evaluate_partseg():
    model_path = Environment.weights_path / "cvpr2024_tetrasphere_partseg_K=8_82_2.t7"
    args = f"--exp_name eval_partseg_K8 --model ts_partseg --num_spheres 8 --model_path {model_path} --test_batch_size 1 --seed 3"

    # This experiment takes a long time to start
    print(f"Evaluating {model_path.stem}")
    print("-------------------------------------")
    eval_partseg(args.split())


if __name__ == "__main__":
    #evaluate_mn40()
    #evaluate_objbg()
    #evaluate_pbt50rs()
    evaluate_partseg()
