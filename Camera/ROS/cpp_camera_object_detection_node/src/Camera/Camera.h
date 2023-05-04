#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>

class Camera : public rclcpp::Node 
{
private:
    sl::Camera zed_;
public:
    Camera();
    ~Camera();

    void CaptureImage();
};