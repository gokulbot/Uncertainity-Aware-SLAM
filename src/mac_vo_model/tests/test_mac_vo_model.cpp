#include <iostream>
#include "model.h"
#include <torch/torch.h>
#include <torch/script.h>

int main() {
    std::cout << "Starting Model Test..." << std::endl;

    // Create a dummy model setup
    ModelSetup setup;
    setup.model_path = "dummy_model.pt"; // Replace with your actual model file path

    // Create Model instance
    Model model;

    // Try to load the model
    if (!model.setup(setup)) {
        std::cerr << "Failed to load model!" << std::endl;
        return -1;
    }

    // Create dummy input tensors
    MACVOInput inputs;
    inputs.left_image        = torch::rand({1, 3, 256, 256});
    inputs.right_image       = torch::rand({1, 3, 256, 256});
    inputs.intrinsics        = torch::eye(3);
    inputs.intrinsics_inv    = torch::eye(3);
    inputs.prev_left_image   = torch::rand({1, 3, 256, 256});
    inputs.prev_right_image  = torch::rand({1, 3, 256, 256});
    inputs.prev_depth        = torch::rand({1, 1, 256, 256});
    inputs.prev_pose         = torch::rand({1, 6});

    MACVOOutput outputs;

    try {
        model.forward(inputs, outputs);
        std::cout << "Model forward pass successful!" << std::endl;
        std::cout << "Depth tensor size: " << outputs.depth.sizes() << std::endl;
        std::cout << "Pose tensor size: " << outputs.pose.sizes() << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error during forward pass: " << e.what() << std::endl;
        return -1;
    }

    std::cout << "Test completed successfully." << std::endl;
    return 0;
}
