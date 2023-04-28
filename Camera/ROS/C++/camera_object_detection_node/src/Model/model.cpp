#include "model.h"

Model::Model(const std::string& filename) 
{
    try {
        model_ = torch::jit::load(filename);
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e << std::endl;
        exit(1);
    }
}

Model::~Model()
{
    model_.reset();
}

void Model::Detect() 
{ }