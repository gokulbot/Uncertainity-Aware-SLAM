#pragma once
#include "setup.h"
#include <torch/script.h> // One-stop header.
#include <memory>
#include "logger.h"

/**
 * @class Model
 * @brief A wrapper class for loading and managing a TorchScript model.
 *
 * The Model class provides functionality to load a TorchScript
 * model from a file path and hold it in memory using a
 * smart pointer for safe resource management.
 */
class Model {

public:
    /**
     * @brief Default constructor for the Model class.
     *
     * Initializes the Model instance. The TorchScript module
     * is not loaded until setup() is called.
     */
    Model();

    /**
     * @brief Destructor for the Model class.
     *
     * Automatically cleans up the loaded TorchScript module
     * when the Model object goes out of scope.
     */
    ~Model();

    /**
     * @brief Loads the TorchScript model from a given setup.
     *
     * @param setup A ModelSetup structure containing the path
     *              to the TorchScript model file.
     *
     * @note This function will terminate the program if the model
     *       cannot be loaded.
     */
    void setup(ModelSetup setup);

    /**
     * @brief Inferes a forward pass through the model.
     * 
     * @param inputs A vector of input tensors to the model.
    * @return A vector of output tensors from the model.
     */
    bool forward(MACVOInput &inputs, MACVOOutput &outputs);

private:
    /**
     * @brief Pointer to the loaded TorchScript module.
     *
     * Stored as a unique_ptr to ensure proper memory management.
     */
    std::unique_ptr<torch::jit::script::Module> mac_vo_module_;
};
