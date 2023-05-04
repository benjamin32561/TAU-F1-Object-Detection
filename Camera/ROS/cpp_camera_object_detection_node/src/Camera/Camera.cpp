#include "Camera.h"

Camera::Camera()
{
    // Initialize the ZED camera
    zed_.open();
    sl::InitParameters init_params;
    init_params.camera_resolution = sl::RESOLUTION::HD720;
    init_params.camera_fps = 30;
    init_params.coordinate_units = sl::UNIT::METER;
    zed_.open(init_params);
}

Camera::~Camera()
{
    // Close the ZED camera
    zed_.close();
}

void Camera::captureAndPublishImage() 
{
    // Capture a new image from the ZED camera
    sl::Mat image;
    zed_.grab();
    zed_.retrieveImage(image, sl::VIEW::LEFT);
}