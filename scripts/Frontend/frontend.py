from __future__ import annotations

import torch
import time
from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass
from typing import Literal

# from Utility.PrettyPrint import Logger
# from Utility.Timer import Timer
from Utility.Utils import reflect_torch_dtype
from Stereo_Depth.stereo_depth_cov import disparity_to_depth, disparity_to_depth_cov

@dataclass
class CUDAGraphHandler:
    graph: torch.cuda.CUDAGraph
    shape: torch.Size
    static_input: dict[str, torch.Tensor]
    static_ouput: dict[str, torch.Tensor]

# -------------------------
# FlowFormerCov Frontend
# -------------------------
class FlowFormerCovFrontend:
    TENSOR_RT_AOT_RESULT_PATH = Path("./cache/FlowFormerCov_TRTCache")
    T_SUPPORT_DTYPE = Literal["fp32", "bf16", "fp16"]

    def __init__(self, config: SimpleNamespace, model=None):
        """
        If model is None, it will be built internally.
        config must contain:
            weight, device, enc_dtype, dec_dtype, enforce_positive_disparity, decoder_depth
        """
        self.config = config

        if model is None:
            from Network.FlowFormer.configs.submission import get_cfg
            from Network.FlowFormerCov import build_flowformer
            cfg = get_cfg()
            cfg.latentcostformer.decoder_depth = self.config.decoder_depth
            model = build_flowformer(cfg, reflect_torch_dtype(config.enc_dtype), reflect_torch_dtype(config.dec_dtype))
            ckpt = torch.load(self.config.weight, map_location=self.config.device, weights_only=True)
            model.load_ddp_state_dict(ckpt)

        self.model = model.to(self.config.device).eval()

    @property
    def provide_cov(self):
        return True, True

    @staticmethod
    def inference_2_depth(flow_12: torch.Tensor, cov_12: torch.Tensor, fx, baseline, enforce_positive_disparity: bool):
        disparity, disparity_cov = flow_12[:, :1].abs(), cov_12[:, :1]
        depth_map = disparity_to_depth(disparity, baseline, fx)
        depth_cov = disparity_to_depth_cov(disparity, disparity_cov, baseline, fx)
        bad_mask = flow_12[:, :1] <= 0 if enforce_positive_disparity else None
        return SimpleNamespace(depth=depth_map, cov=depth_cov, disparity=disparity, disparity_uncertainty=disparity_cov, mask=bad_mask)

    @staticmethod
    def inference_2_match(flow_12: torch.Tensor, cov_12: torch.Tensor):
        return SimpleNamespace(flow=flow_12, cov=cov_12, mask=None)

    @torch.inference_mode()
    def estimate_depth(self, imageL, imageR, fx, baseline):
        est_flow, est_cov = self.model.inference(imageL.to(self.config.device), imageR.to(self.config.device))
        return self.inference_2_depth(est_flow, est_cov, fx, baseline, self.config.enforce_positive_disparity)

    # @Timer.cpu_timeit("Frontend.estimate")
    # @Timer.gpu_timeit("Frontend.estimate")
    @torch.inference_mode()
    def estimate_pair(self, imageL1, imageL2, imageR1, imageR2, fx, baseline):
        input_A = torch.cat([imageL2, imageL1], dim=0)
        input_B = torch.cat([imageR2, imageL2], dim=0)
        input_A = input_A.to(self.config.device)
        input_B = input_B.to(self.config.device)
        est_flow, est_cov = self.model.inference(input_A, input_B)
        return (
            self.inference_2_depth(est_flow[0:1], est_cov[0:1], fx, baseline, self.config.enforce_positive_disparity),
            self.inference_2_match(est_flow[1:2], est_cov[1:2])
        )

    @torch.inference_mode()
    def estimate_triplet(self, imageL1, imageL2, imageR1, imageR2, fx, baseline):
        input_A = torch.cat([imageL1, imageL2, imageL1], dim=0)
        input_B = torch.cat([imageL1, imageR2, imageL2], dim=0)
        input_A = input_A.to(self.config.device)
        input_B = input_B.to(self.config.device)
        est_flow, est_cov = self.model.inference(input_A, input_B)
        return (
            self.inference_2_depth(est_flow[0:1], est_cov[0:1], fx, baseline, self.config.enforce_positive_disparity),
            self.inference_2_depth(est_flow[1:2], est_cov[1:2], fx, baseline, self.config.enforce_positive_disparity),
            self.inference_2_match(est_flow[2:3], est_cov[2:3])
        )

# -------------------------
# CUDAGraph FlowFormerCov Frontend
# -------------------------
class CUDAGraph_FlowFormerCovFrontend(FlowFormerCovFrontend):
    def __init__(self, config: SimpleNamespace, model=None):
        super().__init__(config, model)
        self.cuda_graph: CUDAGraphHandler | None = None
        assert "cuda" in self.config.device.lower(), "CUDAGraph frontend requires CUDA"
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("medium")
        torch.backends.cuda.preferred_linalg_library = "cusolver"

    # @Timer.cpu_timeit("Frontend.estimate")
    # @Timer.gpu_timeit("Frontend.estimate")
    def estimate_pair(self, imageL1, imageL2, imageR1, imageR2, fx, baseline):
        input_A = torch.cat([imageL2, imageL1], dim=0).to(self.config.device)
        input_B = torch.cat([imageR2, imageR1], dim=0).to(self.config.device)
        est_flow, est_cov = self.cuda_graph_estimate(input_A, input_B)
        return (
            self.inference_2_depth(est_flow[0:1], est_cov[0:1], fx, baseline, self.config.enforce_positive_disparity),
            self.inference_2_match(est_flow[1:2], est_cov[1:2])
        )

    def cuda_graph_estimate(self, inp_A: torch.Tensor, inp_B: torch.Tensor):
        if self.cuda_graph is None:
            # Logger.write("info", "Building CUDAGraph for FlowFormerCovFrontend")
            static_input_A, static_input_B = torch.empty_like(inp_A), torch.empty_like(inp_B)
            static_input_A.copy_(inp_A)
            static_input_B.copy_(inp_B)

            for _ in range(3):  # warm-up
                output_val, output_cov = self.model.inference(static_input_A, static_input_B)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_output, static_output_cov = self.model.inference(static_input_A, static_input_B)

            self.cuda_graph = CUDAGraphHandler(
                graph, inp_A.shape,
                static_input={"input_A": static_input_A, "input_B": static_input_B},
                static_ouput={"flow": static_output, "flow_cov": static_output_cov}
            )
            return output_val, output_cov
        else:
            g_context = self.cuda_graph
            g_context.static_input["input_A"].copy_(inp_A)
            g_context.static_input["input_B"].copy_(inp_B)
            g_context.graph.replay()
            return g_context.static_ouput["flow"].clone(), g_context.static_ouput["flow_cov"].clone()
