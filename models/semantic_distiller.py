"""OPT-22: Semantic feature distillation from SAM2 to Gaussian attributes"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticFeatureDistiller(nn.Module):
    """Distills SAM2 multi-scale features into per-Gaussian semantic vectors.
    Enables semantic rendering and boundary refinement through the 4DGS pipeline.
    
    Key innovation: Features are not just stored but actively used during
    rasterization to enforce semantic consistency across views."""

    def __init__(self, sam2_feat_dim=256, gaussian_sem_dim=64, n_classes=10):
        super().__init__()
        # Project SAM2 features to compact Gaussian semantic space
        self.feat_projector = nn.Sequential(
            nn.Linear(sam2_feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, gaussian_sem_dim)
        )
        # Semantic classifier head
        self.classifier = nn.Linear(gaussian_sem_dim, n_classes)
        # Boundary refinement head
        self.boundary_head = nn.Sequential(
            nn.Linear(gaussian_sem_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.gaussian_sem_dim = gaussian_sem_dim

    def forward(self, sam2_features, gaussian_ids):
        """Project SAM2 features to per-Gaussian semantic vectors.
        Args:
            sam2_features: [N, sam2_feat_dim] extracted SAM2 features
            gaussian_ids: [M] indices mapping Gaussians to feature vectors
        Returns:
            sem_vectors: [M, gaussian_sem_dim] semantic attribute vectors
            class_logits: [M, n_classes] classification logits
            boundary_probs: [M, 1] boundary probability
        """
        # Project to compact space
        compact_feats = self.feat_projector(sam2_features)
        sem_vectors = compact_feats[gaussian_ids]

        # Classification
        class_logits = self.classifier(sem_vectors)

        # Boundary detection
        boundary_probs = self.boundary_head(sem_vectors)

        return sem_vectors, class_logits, boundary_probs

    def semantic_render_loss(self, rendered_sem, gt_sem, boundary_probs, boundary_gt):
        """Combined semantic rendering loss.
        Args:
            rendered_sem: [H, W, n_classes] rendered semantic map
            gt_sem: [H, W, n_classes] ground truth semantic labels
            boundary_probs: [M, 1] predicted boundary probabilities
            boundary_gt: [M, 1] boundary ground truth from SAM2
        """
        # Cross-entropy for semantic classification
        sem_loss = F.cross_entropy(rendered_sem.permute(2, 0, 1).unsqueeze(0),
                                    gt_sem.argmax(dim=-1).unsqueeze(0).long())
        # Boundary BCE loss with hard negative mining
        bce = F.binary_cross_entropy(boundary_probs, boundary_gt, reduction='none')
        # Focus on hard examples (boundary regions)
        hard_mask = (boundary_gt > 0.5) | (boundary_probs > 0.5)
        boundary_loss = bce[hard_mask].mean() if hard_mask.any() else bce.mean()

        return sem_loss + 0.5 * boundary_loss
