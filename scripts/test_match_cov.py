# test_flowformer_cov_matcher.py

import torch
from types import SimpleNamespace
from Matching.matcher import FlowFormerCovMatcher  # adjust path if needed

def test_flowformer_cov_matcher():
    # ---- 1. Create dummy images ----
    B, C, H, W = 1, 3, 256, 320
    image1 = torch.rand(B, C, H, W)
    image2 = torch.rand(B, C, H, W)

    # ---- 2. Camera / model config ----
    config = SimpleNamespace(
        weight="/work/models/MACVO_FrontendCov.pth",  # <-- replace with real checkpoint
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        enc_dtype="fp16",
        dec_dtype="fp16",
    )

    # ---- 3. Validate config ----
    FlowFormerCovMatcher.is_valid_config(config)

    # ---- 4. Instantiate matcher ----
    matcher = FlowFormerCovMatcher(config)

    # ---- 5. Run estimation ----
    output = matcher.estimate(image1, image2)

    # ---- 6. Print shapes ----
    print("Flow shape:", output.flow.shape)      # should be Bx2xHxW
    print("Cov shape:", output.cov.shape)        # should be Bx3xHxW
    print("Mask:", output.mask)                  # None for now

    # ---- 7. Optionally inspect a sample ----
    print("Sample flow [0,0,:,:]:")
    print(output.flow[0, 0])

    print("Sample cov [0,0,:,:]:")
    print(output.cov[0, 0])

if __name__ == "__main__":
    test_flowformer_cov_matcher()
