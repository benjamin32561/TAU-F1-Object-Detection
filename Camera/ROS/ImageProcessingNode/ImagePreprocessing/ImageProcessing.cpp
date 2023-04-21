#include "ImagePreprocessing.h"

ImagePreprocessing::ImagePreprocessing() 
{ }

sensor_msgs::msg::Image::SharedPtr ImagePreprocessing::preprocess(const sensor_msgs::msg::Image::SharedPtr& msg) 
{
    // Convert the image message to an OpenCV Mat
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR_STREAM(rclcpp::get_logger("ImagePreprocessing"), "cv_bridge exception: " << e.what());
        return nullptr;
    }

    // Apply any necessary image processing here
    cv::Mat processed_image;
    cv::cvtColor(cv_ptr->image, processed_image, cv::COLOR_BGR2GRAY);

    // Convert the processed image back to a sensor_msgs/Image message
    sensor_msgs::msg::Image::SharedPtr processed_msg = cv_bridge::CvImage(std_msgs::msg::Header(),
                                                                            sensor_msgs::image_encodings::MONO8,
                                                                            processed_image).toImageMsg();

    return processed_msg;
}