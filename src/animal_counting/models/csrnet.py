from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as TF

from animal_counting.datasets.base import BaseAnimalCountingDataset
from animal_counting.datasets.density_map import DensityMapDataset
from animal_counting.models.base import BaseCountingModel, CountingParadigm, PredictionResult


class CSRNet(nn.Module):
    """
    CSRNet: Dilated Convolutional Neural Networks for Understanding the
    Highly Congested Scenes (Li et al., CVPR 2018).

    Frontend: VGG-16 layers 0-22 (conv1_1 through conv4_3, no pool4).
              Spatial stride = 8 (three max-pool operations: pool1, pool2, pool3).
    Backend:  Six dilated 3×3 convolutions (dilation=2) followed by a 1×1
              projection to produce a single-channel density map.
    Output:   Density map at 1/8 the input resolution.
              Predicted count = density_map.sum().
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        vgg = models.vgg16(
            weights=models.VGG16_Weights.DEFAULT if pretrained else None
        )
        # VGG-16 feature layer indices:
        #  0-4  : conv1_1, conv1_2, pool1   (stride 2)
        #  5-9  : conv2_1, conv2_2, pool2   (stride 4)
        # 10-16 : conv3_1..3, pool3         (stride 8)
        # 17-22 : conv4_1..3  (NO pool4 → still stride 8)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])

        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frontend(x)
        x = self.backend(x)
        return x  # (B, 1, H/8, W/8)


class CSRNetCountingModel(BaseCountingModel):
    """
    Density-map counting model wrapping CSRNet.

    Training uses MSE loss between the predicted density map and a ground-truth
    map generated from point annotations (or bounding-box centres for box-only
    datasets) with adaptive Gaussian kernels.

    Count is inferred as the integral (sum) of the predicted density map.
    """

    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, device: str | None = None, pretrained: bool = True) -> None:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(name="CSRNet", paradigm=CountingParadigm.DENSITY_MAP, device=device)
        self.net = CSRNet(pretrained=pretrained).to(self.device)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_dataset: BaseAnimalCountingDataset,
        val_dataset: BaseAnimalCountingDataset,
        epochs: int = 200,
        batch_size: int = 8,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        patch_size: int = 512,
        density_scale: int = 8,
        beta: float = 0.3,
        k: int = 3,
        patience: int = 20,
        output_dir: str | Path | None = None,
        num_workers: int = 4,
        resume: bool = True,
        **kwargs: Any,
    ) -> Mapping[str, float]:
        """
        Train CSRNet with MSE loss on density maps.

        Args:
            train_dataset: Any BaseAnimalCountingDataset (split='train').
            val_dataset:   Any BaseAnimalCountingDataset (split='val').
            epochs:        Maximum training epochs.
            batch_size:    Images per gradient step.
            lr:            Adam learning rate.
            weight_decay:  Adam weight decay.
            patch_size:    Square patch size sampled from each image.
            density_scale: Spatial downsampling factor for the density map
                           (must match the network's frontend stride = 8).
            beta:          Gaussian sigma scale factor (sigma = beta * mean_k_dist).
            k:             Number of nearest neighbours for sigma estimation.
            patience:      Early-stopping patience (epochs without val MAE improvement).
            output_dir:    Directory where best.pth and checkpoints are written.
            num_workers:   DataLoader worker processes.
            resume:        If True, resume from checkpoint if one exists.
        """
        output_dir = Path(output_dir) if output_dir else Path("results/csrnet")
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        train_dm = DensityMapDataset(
            train_dataset, patch_size=patch_size, augment=True,
            density_scale=density_scale, beta=beta, k=k,
        )
        val_dm = DensityMapDataset(
            val_dataset,
            patch_size=patch_size,
            augment=False,
            full_image=True,
            density_scale=density_scale,
            beta=beta,
            k=k,
        )

        train_loader = DataLoader(
            train_dm, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_dm, batch_size=1, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=lr, weight_decay=weight_decay
        )

        best_mae = float("inf")
        no_improve = 0
        start_epoch = 1

        # Try to resume from checkpoint
        latest_checkpoint = self._find_latest_checkpoint(checkpoint_dir)
        if resume and latest_checkpoint:
            print(f"Resuming from checkpoint: {latest_checkpoint}")
            checkpoint = self.load_checkpoint(latest_checkpoint)
            start_epoch = checkpoint["epoch"] + 1
            best_mae = checkpoint["best_mae"]
            no_improve = checkpoint["no_improve"]
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            print(f"  Resuming from epoch {start_epoch} (best_mae={best_mae:.2f}, no_improve={no_improve})")

        for epoch in range(start_epoch, epochs + 1):
            train_loss = self._train_epoch(train_loader, optimizer)
            val_mae, val_rmse = self._validate(val_loader)

            print(
                f"Epoch {epoch:>3}/{epochs} | "
                f"train_loss={train_loss:.6f} | "
                f"val_MAE={val_mae:.2f} | val_RMSE={val_rmse:.2f}"
            )

            if val_mae < best_mae:
                best_mae = val_mae
                no_improve = 0
                self.save(output_dir / "best.pth")
                print(f"  → New best val MAE: {best_mae:.2f}  (saved to {output_dir / 'best.pth'})")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                    break

            # Save checkpoint after every 5 epochs
            if epoch % 5 == 0:
                self.save_checkpoint(
                    checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth",
                    epoch=epoch,
                    best_mae=best_mae,
                    no_improve=no_improve,
                    optimizer_state=optimizer.state_dict(),
                )

                # delete previous checkpoints to save space
                self._cleanup_checkpoints(checkpoint_dir, keep_last_n=2)

        self.load(output_dir / "best.pth")
        print(f"Training complete. Best val MAE: {best_mae:.2f}")
        
        # Clean up checkpoints after successful completion
        self._cleanup_checkpoints(checkpoint_dir)
        
        return {"best_val_mae": best_mae}

    def _train_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        self.net.train()
        total_loss = 0.0
        for images, density_maps in loader:
            images = images.to(self.device)
            density_maps = density_maps.to(self.device)

            pred = self.net(images)
            loss = F.mse_loss(pred, density_maps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        return total_loss / len(loader.dataset)

    def _validate(self, loader: DataLoader) -> tuple[float, float]:
        mae_sum = 0.0
        mse_sum = 0.0
        n = 0

        self.net.eval()
        with torch.no_grad():
            for images, density_maps in loader:
                images = images.to(self.device)
                density_maps = density_maps.to(self.device)

                pred = self.net(images)
                pred_count = pred.sum().item()
                gt_count = density_maps.sum().item()

                err = abs(float(pred_count) - float(gt_count))
                mae_sum += err
                mse_sum += err ** 2
                n += 1

        mae = mae_sum / n
        rmse = (mse_sum / n) ** 0.5
        return mae, rmse

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, image: Any, **kwargs: Any) -> PredictionResult:
        """
        Run inference on a single image.

        Args:
            image: PIL Image, H×W×3 uint8 numpy array, or CHW float32 tensor.

        Returns:
            PredictionResult with ``count`` (float) and ``density_map`` (1×H/8×W/8 tensor).
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))

        if isinstance(image, Image.Image):
            img_tensor = TF.to_tensor(image)
        else:
            img_tensor = image  # assume CHW tensor

        img_tensor = TF.normalize(img_tensor, mean=self._IMAGENET_MEAN, std=self._IMAGENET_STD)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        self.net.eval()
        with torch.no_grad():
            density_map = self.net(img_tensor)  # (1, 1, H/8, W/8)

        density_map = density_map.squeeze(0).cpu()  # (1, H/8, W/8)
        count = float(density_map.sum().item())

        return PredictionResult(count=count, density_map=density_map)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path)

    def load(self, path: str | Path) -> None:
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(state_dict)

    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int,
        best_mae: float,
        no_improve: int,
        optimizer_state: dict,
    ) -> None:
        """Save full training checkpoint with model, optimizer, and training state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "model_state": self.net.state_dict(),
            "optimizer_state": optimizer_state,
            "best_mae": best_mae,
            "no_improve": no_improve,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str | Path) -> dict:
        """Load full training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint["model_state"])
        return checkpoint

    @staticmethod
    def _find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
        """Find the latest checkpoint by epoch number."""
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            return None
        
        checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        if not checkpoints:
            return None
        
        # Sort by epoch number and return the latest
        checkpoints.sort(
            key=lambda p: int(p.stem.split("_")[-1]),
            reverse=True
        )
        return checkpoints[0]

    @staticmethod
    def _cleanup_checkpoints(checkpoint_dir: Path, keep_last_n: int = 3) -> None:
        """Remove old checkpoints, keeping only the last N."""
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            return
        
        checkpoints = sorted(
            checkpoint_dir.glob("checkpoint_epoch_*.pth"),
            key=lambda p: int(p.stem.split("_")[-1]),
            reverse=True
        )
        
        # Remove all but the last keep_last_n checkpoints
        for checkpoint in checkpoints[keep_last_n:]:
            checkpoint.unlink()
            print(f"Cleaned up: {checkpoint.name}")
