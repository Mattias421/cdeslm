import sys
from pathlib import Path
from typing import Optional, TextIO
import equinox as eqx
import jax


class Logger:
    def __init__(
        self,
        log_file: Optional[Path] = None,
        loss_file: Optional[Path] = None,
        valid_loss_file: Optional[Path] = None,
    ):
        self.log_file = log_file
        self.console = sys.stdout
        self.file_handle: Optional[TextIO] = None
        self.loss_file_handle: Optional[TextIO] = None
        self.valid_loss_file_handle: Optional[TextIO] = None

        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            self.file_handle = open(log_file, "a", buffering=1)

        if loss_file:
            loss_file.parent.mkdir(parents=True, exist_ok=True)
            self.loss_file_handle = open(loss_file, "a", buffering=1)

        if valid_loss_file:
            valid_loss_file.parent.mkdir(parents=True, exist_ok=True)
            self.valid_loss_file_handle = open(valid_loss_file, "a", buffering=1)

    def _write(self, message: str):
        print(message, file=self.console, flush=True)
        if self.file_handle:
            print(message, file=self.file_handle, flush=True)

    def close(self):
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
        if self.loss_file_handle:
            self.loss_file_handle.close()
            self.loss_file_handle = None
        if self.valid_loss_file_handle:
            self.valid_loss_file_handle.close()
            self.valid_loss_file_handle = None

    def info(self, message: str):
        self._write(f"[INFO] {message}")

    def train(self, step: int, loss: float):
        msg = f"[TRAIN] Step: {step:6d} | Loss: {loss:.6f}"
        print(msg, file=self.console, flush=True)
        if self.file_handle:
            print(msg, file=self.file_handle, flush=True)
        if self.loss_file_handle:
            print(f"{step}\t{loss:.6f}", file=self.loss_file_handle, flush=True)

    def valid(self, step: int, loss: float):
        msg = f"[VALID] Step: {step:6d} | Loss: {loss:.6f}"
        print(msg, file=self.console, flush=True)
        if self.file_handle:
            print(msg, file=self.file_handle, flush=True)
        if self.valid_loss_file_handle:
            print(f"{step}\t{loss:.6f}", file=self.valid_loss_file_handle, flush=True)

    def config(self, **kwargs):
        self.info("=== TRAINING CONFIGURATION ===")
        for key, value in kwargs.items():
            self.info(f"{key}: {value}")
        self.info("=" * 40)

    def model_summary(self, model: eqx.Module):
        self.info("=== MODEL ARCHITECTURE ===")
        params = eqx.filter(model, eqx.is_array)
        param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
        self.info(f"Total parameters: {param_count:,}")
        self.info(f"Initial MLP: {model.initial}")
        self.info(f"CDE Function MLP: {model.func.mlp}")
        self.info(f"Readout: {model.readout}")
        self.info("=" * 40)


def setup_logger(output_dir: Optional[str] = None) -> Logger:
    log_file = Path(output_dir) / "train.log" if output_dir else None
    loss_file = Path(output_dir) / "loss.tsv" if output_dir else None
    valid_loss_file = Path(output_dir) / "valid_loss.tsv" if output_dir else None
    return Logger(log_file, loss_file, valid_loss_file)
