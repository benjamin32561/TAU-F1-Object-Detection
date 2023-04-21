cmake_minimum_required(VERSION 3.5)
project(camera_object_detection_node)

# Find required packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

# Add the executable
add_executable(camera_object_detection_node src/camera_object_detection_node.cpp)

# Link against required libraries
target_link_libraries(camera_object_detection_node
  ${rclcpp_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
  ${cv_bridge_LIBRARIES}
  ${image_transport_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${TORCH_LIBRARIES}
)

# Install the executable
install(TARGETS
  camera_object_detection_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch file
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install package.xml
install(FILES
  package.xml
  DESTINATION share/${PROJECT_NAME}/
)

# Export dependencies
ament_export_dependencies(rclcpp sensor_msgs cv_bridge image_transport)

# Build type, C++ standard
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 14)

# Build with optimization and all warnings
if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  add_compile_options(-O3 -Wall -Wextra)
endif()

# Install package
ament_package()