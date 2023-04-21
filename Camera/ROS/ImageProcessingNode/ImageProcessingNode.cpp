#include "ImagePreprocessingNode.h"

ImagePreprocessingNode::ImagePreprocessingNode() : Node("image_preprocessing_node") 
{
    // Create the subscriber for the image topic
    this->image_sub_ = create_subscription<sensor_msgs::msg::Image>("image_topic", 10,
                                                                std::bind(&ImagePreprocessingNode::processImage, this, std::placeholders::_1));

    // Create the publisher for the processed image topic
    this->processed_image_pub_ = create_publisher<sensor_msgs::msg::Image>("processed_image", 10);

    // Initialize the ImagePreprocessing class
    this->image_preprocessing_ = std::make_shared<ImagePreprocessing>();
}


void ImagePreprocessingNode::processImage(const sensor_msgs::msg::Image::SharedPtr& msg) 
{
    // Preprocess the image data
    sensor_msgs::msg::Image::SharedPtr processed_msg = this->image_preprocessing_->preprocess(msg);

    if (processed_msg != nullptr) {
        // Publish the processed image data
        this->processed_image_pub_->publish(*processed_msg);
    }
}