#include <torch/script.h>

class BaseModel 
{
private:
    std::shared_ptr<torch::jit::script::Module> model_;
public:
  virtual void DetectObjects() = 0; // Pure virtual function
  virtual int calculate(int x, int y) { // Virtual function with default implementation
    return x + y;
  }
};