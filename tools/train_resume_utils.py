import random

import numpy as np
import torch
from torch.utils.data import Sampler


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def capture_rng_states():
    return {
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "rng_numpy": np.random.get_state(),
        "rng_python": random.getstate(),
    }


def restore_rng_states(state_dict):
    torch.set_rng_state(state_dict["rng_torch"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state_dict["rng_cuda"])
    np.random.set_state(state_dict["rng_numpy"])
    random.setstate(state_dict["rng_python"])


def require_ckpt_keys(state_dict, required_keys, ckpt_path):
    missing = [key for key in required_keys if key not in state_dict]
    if missing:
        raise KeyError(f"checkpoint {ckpt_path} missing required keys: {missing}")


class StatefulRandomSampler(Sampler):
    def __init__(self, data_source, seed = 0):
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0
        self.start_index = 0
        self.indices = None

    def __len__(self):
        return len(self.data_source)

    def _build_epoch_indices(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        self.indices = torch.randperm(len(self.data_source), generator = generator).tolist()

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self.start_index = 0
        self._build_epoch_indices()

    def set_start_step(self, step_in_epoch, batch_size):
        self.start_index = max(0, int(step_in_epoch) * int(batch_size))

    def __iter__(self):
        if self.indices is None:
            self._build_epoch_indices()
        start = min(self.start_index, len(self.indices))
        return iter(self.indices[start:])

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "start_index": self.start_index,
            "seed": self.seed,
        }

    def load_state_dict(self, state_dict):
        require_ckpt_keys(state_dict, ["epoch", "start_index", "seed"], "sampler_state")
        self.epoch = int(state_dict["epoch"])
        self.start_index = int(state_dict["start_index"])
        self.seed = int(state_dict["seed"])
        self._build_epoch_indices()
