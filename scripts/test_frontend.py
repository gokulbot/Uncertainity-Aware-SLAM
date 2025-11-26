# test_frontend_triplet.py

import torch
from types import SimpleNamespace

from Frontend.frontend import FlowFormerCovFrontend


def test_flowformer_cov_frontend_triplet():
    """
    Simple test for FlowFormerCovFrontend.estimate_triplet using random tensors.
    """

    # ---- 1. Dummy stereo sequence: t1 and t2 ----
    B, C, H, W = 1, 3, 256, 320
    imageL1 = torch.rand(B, C, H, W)
    imageR1 = torch.rand(B, C, H, W)
    imageL2 = torch.rand(B, C, H, W)
    imageR2 = torch.rand(B, C, H, W)

    # ---- 2. Camera parameters ----
    baseline = 0.1  # meters
    fx = 500.0      # pixels

    # ---- 3. Frontend configuration ----
    config = SimpleNamespace(
        weight="/work/models/MACVO_FrontendCov.pth",  # <-- make sure this exists
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        enc_dtype="fp16",
        dec_dtype="fp16",
        enforce_positive_disparity=True,
        decoder_depth=3,
    )

    # ---- 4. Instantiate frontend ----
    frontend = FlowFormerCovFrontend(config)

    # ---- 5. Run estimate_triplet ----
    depth_t1, depth_t2, match_t12 = frontend.estimate_triplet(
        imageL1=imageL1,
        imageL2=imageL2,
        imageR1=imageR1,
        imageR2=imageR2,
        fx=fx,
        baseline=baseline,
    )

    # ---- 6. Assertions on depth outputs ----
    for name, out in [("t1", depth_t1), ("t2", depth_t2)]:
        assert out.depth.shape == (B, 1, H, W), f"Depth shape mismatch for {name}"
        assert out.disparity.shape == (B, 1, H, W), f"Disparity shape mismatch for {name}"
        assert out.cov.shape == (B, 1, H, W), f"Depth covariance shape mismatch for {name}"
        assert out.disparity_uncertainty.shape == (B, 1, H, W), f"Disparity uncertainty shape mismatch for {name}"

    # ---- 7. Assertions on match output ----
    assert match_t12.flow.ndim == 4, "Flow must be 4D (B,2,H,W)"
    assert match_t12.cov.ndim == 4, "Flow covariance must be 4D (B,2,H,W)"
    assert match_t12.flow.shape[0] == B, "Flow batch size mismatch"
    assert match_t12.flow.shape[2:] == (H, W), "Flow spatial size mismatch"

    print("Triplet frontend test passed!")
    print("Depth t1:  min/max =", depth_t1.depth.min().item(), depth_t1.depth.max().item())
    print("Depth t2:  min/max =", depth_t2.depth.min().item(), depth_t2.depth.max().item())
    print("Flow t1->t2: min/max =", match_t12.flow.min().item(), match_t12.flow.max().item())

    # ---- 8. Print all shapes ----
    print("Shapes of outputs:")
    print("Depth t1: ", depth_t1.depth.shape)
    print("Disparity t1: ", depth_t1.disparity.shape)
    print("Covariance t1: ", depth_t1.cov.shape)
    print("Disparity Uncertainty t1: ", depth_t1.disparity_uncertainty.shape)
    print("Depth t2: ", depth_t2.depth.shape)
    print("Disparity t2: ", depth_t2.disparity.shape)
    print("Covariance t2: ", depth_t2.cov.shape)
    print("Disparity Uncertainty t2: ", depth_t2.disparity_uncertainty.shape)
    print("Flow t1->t2: ", match_t12.flow.shape)
    print("Flow Covariance t1->t2: ", match_t12.cov.shape)


if __name__ == "__main__":
    test_flowformer_cov_frontend_triplet()
