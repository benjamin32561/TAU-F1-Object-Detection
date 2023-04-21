#include <torch/script.h>

class Model {
    private:
        std::shared_ptr<torch::jit::script::Module> model_;

    public:
        Model(const std::string& filename);
        ~Model();

        void detect();
};
