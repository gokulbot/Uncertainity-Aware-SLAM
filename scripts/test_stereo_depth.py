# test_flowformer_cov_depth.py

import torch
from types import SimpleNamespace
from Stereo_Depth.stereo_depth_cov import FlowFormerCovDepth


def test_flowformer_cov_depth():
    """
    Simple test for FlowFormerCovDepth using random image tensors.
    """

    # ---- 1. Dummy stereo images ----
    B, C, H, W = 1, 3, 256, 320
    imageL = torch.rand(B, C, H, W)
    imageR = torch.rand(B, C, H, W)

    # ---- 2. Camera parameters ----
    baseline = 0.1  # meters
    fx = 500.0      # pixels

    # ---- 3. Model configuration ----
    config = SimpleNamespace(
        weight="/work/models/MACVO_FrontendCov.pth",  # replace with actual path
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        enc_dtype="fp16",
        dec_dtype="fp16",
    )

    # ---- 4. Instantiate model ----
    model = FlowFormerCovDepth(config)

    # ---- 5. Run estimation ----
    output = model.estimate(imageL, imageR, baseline, fx)

    # ---- 6. Assertions & prints ----
    assert output.depth.shape == (B, 1, H, W), "Depth shape mismatch"
    assert output.disparity.shape == (B, 1, H, W), "Disparity shape mismatch"
    assert output.cov.shape == (B, 1, H, W), "Depth covariance shape mismatch"
    assert output.disparity_uncertainty.shape == (B, 1, H, W), "Disparity uncertainty shape mismatch"

    print("Test passed!")
    print("Depth sample [0,0,:,:]:")
    print(output.depth[0, 0])
    print("Disparity sample [0,0,:,:]:")
    print(output.disparity[0, 0])


if __name__ == "__main__":
    test_flowformer_cov_depth()
