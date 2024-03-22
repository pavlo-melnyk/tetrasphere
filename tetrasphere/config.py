from pathlib import Path


class Environment:

    ws_path = Path.home() / "workspace/tetrasphere"
    dset_path = ws_path / "datasets"
    log_path = (ws_path / "training_logs").resolve()
    log_path.mkdir(exist_ok=True, parents=True)
    weights_path = Path(__file__).absolute().parent / "weights"

    wandb_api_key = Path.home() / ".ssh/tetrasphere_wandb_key"
