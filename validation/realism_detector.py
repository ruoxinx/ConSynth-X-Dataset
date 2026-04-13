#!/usr/bin/env python3
"""
UniversalFakeDetect (UnivFD) wrapper for realism validation.

Uses frozen CLIP:ViT-L/14 features + trained FC classifier to score images
as real vs AI-generated. Lower "fake probability" = more realistic augmentation.

Reference:
  Ojha et al. (2023) "Towards Universal Fake Image Detectors that Generalize
  Across Generative Models." CVPR. ArXiv: 2302.10174
  GitHub: https://github.com/WisconsinAIVision/UniversalFakeDetect
  License: MIT

Usage:
  from validation.realism_detector import RealismDetector
  detector = RealismDetector("path/to/fc_weights.pth")
  score = detector.score_image(pil_image)  # 0.0 = real, 1.0 = fake
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


class UnivFDClassifier(nn.Module):
    """Linear classifier on top of frozen CLIP features (UnivFD architecture)."""

    def __init__(self, clip_model, fc_weights_path: str):
        super().__init__()
        self.clip_model = clip_model
        # Freeze CLIP
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Load trained FC layer
        state_dict = torch.load(fc_weights_path, map_location="cpu")
        # UnivFD fc_weights.pth contains a single linear layer: (feature_dim, 1)
        # Detect shape from state dict
        if "weight" in state_dict:
            feat_dim = state_dict["weight"].shape[1]
            self.fc = nn.Linear(feat_dim, 1)
            self.fc.load_state_dict(state_dict)
        elif "fc.weight" in state_dict:
            feat_dim = state_dict["fc.weight"].shape[1]
            self.fc = nn.Linear(feat_dim, 1)
            self.fc.load_state_dict(
                {k.replace("fc.", ""): v for k, v in state_dict.items() if k.startswith("fc.")}
            )
        else:
            # Try loading full model state dict
            feat_dim = 768  # ViT-L/14 default
            self.fc = nn.Linear(feat_dim, 1)
            self.fc.load_state_dict(state_dict)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return fake probability (sigmoid output) for each image."""
        features = self.clip_model.encode_image(images)
        features = features.float()
        features = F.normalize(features, dim=1)
        logits = self.fc(features)
        return torch.sigmoid(logits).squeeze(-1)


class RealismDetector:
    """High-level interface for AI-generated image detection.

    Supports two modes:
      1. UnivFD (CLIP + FC) — requires fc_weights.pth download
      2. CLIP-only (zero-shot) — no extra weights needed, uses text prompts
         "a real photograph" vs "an AI generated image"
    """

    def __init__(
        self,
        fc_weights_path: Optional[str] = None,
        device: str = "cuda",
        mode: str = "auto",
    ):
        """
        Args:
            fc_weights_path: Path to UnivFD fc_weights.pth. If None, uses CLIP zero-shot.
            device: "cuda" or "cpu"
            mode: "univfd" | "clip_zeroshot" | "auto"
                  auto = univfd if weights exist, else clip_zeroshot
        """
        self.device = device

        # Resolve mode
        if mode == "auto":
            if fc_weights_path and Path(fc_weights_path).exists():
                mode = "univfd"
            else:
                mode = "clip_zeroshot"
                if fc_weights_path:
                    print(f"[RealismDetector] Weights not found at {fc_weights_path}, "
                          f"falling back to CLIP zero-shot mode.")

        self.mode = mode
        print(f"[RealismDetector] Mode: {self.mode}")

        # Load CLIP
        import clip
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=device)
        self.clip_model.eval()

        if self.mode == "univfd":
            self.classifier = UnivFDClassifier(self.clip_model, fc_weights_path)
            self.classifier.eval().to(device)
            print(f"[RealismDetector] Loaded UnivFD weights from {fc_weights_path}")
        elif self.mode == "clip_zeroshot":
            # Pre-compute text features for zero-shot classification.
            # Multiple prompts per class for more robust classification.
            real_prompts = [
                "a photo",
                "a real photo",
                "a photograph taken by a camera",
                "a natural photograph",
                "a real image captured outdoors",
            ]
            fake_prompts = [
                "a fake image",
                "an AI generated image",
                "a digitally manipulated image",
                "a synthetic image created by a computer",
                "an artificially generated picture",
            ]
            with torch.no_grad():
                real_tokens = clip.tokenize(real_prompts).to(device)
                fake_tokens = clip.tokenize(fake_prompts).to(device)
                real_feats = self.clip_model.encode_text(real_tokens).float()
                fake_feats = self.clip_model.encode_text(fake_tokens).float()
                # Average across prompts for each class
                self.text_features = torch.stack([
                    F.normalize(real_feats.mean(dim=0), dim=0),
                    F.normalize(fake_feats.mean(dim=0), dim=0),
                ])
                # Use model's learned logit scale
                self.logit_scale = self.clip_model.logit_scale.exp().item()
            print(f"[RealismDetector] CLIP zero-shot ready (logit_scale={self.logit_scale:.1f})")

    @torch.no_grad()
    def _preprocess_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """Preprocess a list of PIL images into a batch tensor."""
        tensors = [self.clip_preprocess(img) for img in images]
        return torch.stack(tensors).to(self.device)

    @torch.no_grad()
    def score_batch(self, images: List[Image.Image]) -> np.ndarray:
        """Score a batch of images. Returns array of fake probabilities (0=real, 1=fake)."""
        batch = self._preprocess_batch(images)

        if self.mode == "univfd":
            scores = self.classifier(batch)
            return scores.cpu().numpy()
        else:
            # CLIP zero-shot
            image_features = self.clip_model.encode_image(batch)
            image_features = F.normalize(image_features.float(), dim=1)
            # Cosine similarity with text prompts
            # text_features[0] = "real", text_features[1] = "AI generated"
            similarity = image_features @ self.text_features.T  # (B, 2)
            # Use model's learned logit scale for calibrated softmax
            probs = F.softmax(similarity * self.logit_scale, dim=1)
            fake_probs = probs[:, 1]
            return fake_probs.cpu().numpy()

    def score_image(self, image: Image.Image) -> float:
        """Score a single image. Returns fake probability (0=real, 1=fake)."""
        return float(self.score_batch([image])[0])

    def score_image_path(self, path: str) -> float:
        """Score an image from file path."""
        img = Image.open(path).convert("RGB")
        return self.score_image(img)


def compute_fooling_rate(
    real_scores: np.ndarray,
    augmented_scores: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute fooling rate and related statistics.

    "Fooling rate" = fraction of augmented images classified as REAL (score < threshold).
    Higher fooling rate = more realistic augmentation.

    Also computes distributional comparison between real and augmented scores.
    """
    real_classified_real = (real_scores < threshold).mean()
    aug_classified_real = (augmented_scores < threshold).mean()

    return {
        "real_accuracy": float(real_classified_real),
        "fooling_rate": float(aug_classified_real),
        "aug_mean_score": float(augmented_scores.mean()),
        "aug_std_score": float(augmented_scores.std()),
        "real_mean_score": float(real_scores.mean()),
        "real_std_score": float(real_scores.std()),
        "score_gap": float(augmented_scores.mean() - real_scores.mean()),
        "threshold": threshold,
        "n_real": len(real_scores),
        "n_augmented": len(augmented_scores),
    }
