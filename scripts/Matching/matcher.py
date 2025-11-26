from __future__ import annotations

import torch
import jaxtyping as Jt
from typeguard import typechecked
from types import SimpleNamespace
from dataclasses import dataclass
from typing import Optional

from Utility.Utils import reflect_torch_dtype

# ----------------------------------------
# FlowFormerCovMatcher Implementation
# ----------------------------------------
class FlowFormerCovMatcher:
    """
    Use modified FlowFormer to jointly estimate optical flow between two frames.
    """

    @Jt.jaxtyped(typechecker=typechecked)
    @dataclass
    class Output:
        flow: Jt.Float32[torch.Tensor, "B 2 H W"]                 # B x 2 x H x W
        cov: Optional[Jt.Float32[torch.Tensor, "B 3 H W"]] = None # B x 3 x H x W, uu, vv, uv
        mask: Optional[Jt.Bool[torch.Tensor, "B 1 H W"]] = None

        @classmethod
        def from_partial_cov(cls,
                             flow: Jt.Float32[torch.Tensor, "B 2 H W"],
                             cov: Jt.Float32[torch.Tensor, "B 2 H W"]) -> "FlowFormerCovMatcher.Output":
            B, C, H, W = cov.shape
            assert C == 2, "Partial cov should have shape Bx2xHxW for uu,vv"
            return cls(flow=flow, cov=torch.cat([cov, torch.zeros((B, 1, H, W), device=cov.device)], dim=1))

    def __init__(self, config: SimpleNamespace):
        self.config = config

        from Network.FlowFormer.configs.submission import get_cfg
        from Network.FlowFormerCov import build_flowformer

        model = build_flowformer(
            get_cfg(),
            encoder_dtype=reflect_torch_dtype(self.config.enc_dtype),
            decoder_dtype=reflect_torch_dtype(self.config.dec_dtype)
        )

        ckpt = torch.load(self.config.weight, weights_only=True)
        model.load_ddp_state_dict(ckpt)
        model.to(self.config.device)
        model.eval()

        self.model = model

    @property
    def provide_cov(self) -> bool:
        return True

    @torch.inference_mode()
    def estimate(self, image1: torch.Tensor, image2: torch.Tensor) -> Output:
        """
        Run inference on two images tensors: Bx3xHxW
        """
        flow, flow_cov = self.model.inference(
            image1.to(self.config.device),
            image2.to(self.config.device)
        )
        return self.Output.from_partial_cov(flow=flow, cov=flow_cov)

    @staticmethod
    def is_valid_config(config: SimpleNamespace) -> None:
        if not isinstance(config.weight, str):
            raise ValueError("weight must be a string")
        if not isinstance(config.device, str) or ("cuda" not in config.device and config.device != "cpu"):
            raise ValueError("device must be cuda or cpu")
        if config.enc_dtype not in {"fp16", "bf16", "fp32"}:
            raise ValueError("enc_dtype must be fp16/bf16/fp32")
        if config.dec_dtype not in {"fp16", "bf16", "fp32"}:
            raise ValueError("dec_dtype must be fp16/bf16/fp32")

