import torch
from types import SimpleNamespace
from typing import TypeVar, Sequence, Callable, Optional, Literal
from Utility.Utils import reflect_torch_dtype
from Utility.Extensions import OnCallCompiler


class FlowFormerCovDepth:
    """
    Estimate disparity and depth uncertainty using FlowFormer.
    """

    def __init__(self, config: SimpleNamespace):
        """
        config should contain:
            weight: path to model checkpoint
            device: "cuda:0" or "cpu"
            enc_dtype: "fp16", "bf16", "fp32"
            dec_dtype: "fp16", "bf16", "fp32"
        """
        from Network.FlowFormer.configs.submission import get_cfg
        from Network.FlowFormerCov import build_flowformer

        model = build_flowformer(
            get_cfg(),
            reflect_torch_dtype(config.enc_dtype),
            reflect_torch_dtype(config.dec_dtype)
        )

        ckpt = torch.load(config.weight, weights_only=True)
        model.load_ddp_state_dict(ckpt)
        model.to(config.device)
        model.eval()

        self.model = model
        self.device = config.device

    @torch.inference_mode()
    def estimate(
        self,
        imageL: torch.Tensor,
        imageR: torch.Tensor,
        baseline: float,
        fx: float
    ) -> SimpleNamespace:
        """
        Args:
            imageL, imageR: [B, 3, H, W] torch tensors
            baseline: stereo baseline (meters)
            fx: focal length (pixels)
        Returns:
            SimpleNamespace containing:
                depth: [B, 1, H, W]
                disparity: [B, 1, H, W]
                cov: [B, 1, H, W] depth covariance
                disparity_uncertainty: [B, 1, H, W]
        """
        est_flow, est_cov = self.model.inference(
            imageL.to(self.device),
            imageR.to(self.device)
        )

        disparity = est_flow[:, :1].abs()
        disparity_cov = est_cov[:, :1]

        depth_map = disparity_to_depth(disparity, baseline, fx)
        depth_cov = disparity_to_depth_cov(disparity, disparity_cov, baseline, fx)

        return SimpleNamespace(
            depth=depth_map,
            disparity=disparity,
            cov=depth_cov,
            disparity_uncertainty=disparity_cov
        )


# ----------------------------------------
# Utility functions
# ----------------------------------------
@OnCallCompiler()
def disparity_to_depth(disp: torch.Tensor, bl: float, fx: float) -> torch.Tensor:
    return (bl * fx) * disp.reciprocal()


@OnCallCompiler()
def disparity_to_depth_cov(disp: torch.Tensor, disp_cov: torch.Tensor, bl: float, fx: float) -> torch.Tensor:
    disparity_2 = disp.square()
    error_rate_2 = disp_cov * disparity_2.reciprocal()
    depth_cov = ((bl * fx) ** 2) * (error_rate_2 / disparity_2)
    return depth_cov
