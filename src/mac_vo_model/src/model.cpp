#include <iostream>
#include "model.h"

/**
 * @class Model
 * @brief Represents a machine learning model using LibTorch.
 *
 * This class encapsulates the functionality for loading and managing
 * a TorchScript model.
 */
Model::Model() {
    // Constructor
}

/**
 * @brief Destructor for the Model class.
 *
 * Cleans up resources associated with the model.
 */
Model::~Model() {
    // Destructor
}

/**
 * @brief Loads and sets up the model from a given configuration.
 *
 * This function deserializes a TorchScript model from a file specified
 * in the `setup` parameter. It uses `torch::jit::load` to load the model.
 *
 * @param setup A ModelSetup structure containing model configuration,
 *              including the path to the model file.
 *
 * @throws Exits the program if the model cannot be loaded.
 */
bool Model::setup(ModelSetup setup) {
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        mac_vo_module_ = std::make_unique<torch::jit::script::Module>(torch::jit::load(setup.model_path));
    }
    catch (const c10::Error& e) {
        LOG_ERROR("error loading the model\n");
        return false;
    }
    LOG_INFO("Model loaded successfully\n");
    return true;
}

bool Model::forward(MACVOInput &inputs, MACVOOutput &outputs) {
    // Prepare input tensors
    // std::vector<torch::jit::IValue> model_inputs;
    // model_inputs.push_back(inputs.left_image);
    // model_inputs.push_back(inputs.right_image);
    // model_inputs.push_back(inputs.intrinsics);
    // model_inputs.push_back(inputs.intrinsics_inv);
    // model_inputs.push_back(inputs.prev_left_image);
    // model_inputs.push_back(inputs.prev_right_image);
    // model_inputs.push_back(inputs.prev_depth);
    // model_inputs.push_back(inputs.prev_pose);

    // // Execute the model and turn its output into a tensor.
    // auto output = mac_vo_module_->forward(model_inputs).toTuple();

    // // Extract outputs
    // outputs.depth = output->elements()[0].toTensor();
    // outputs.pose = output->elements()[1].toTensor();

    return false;
}