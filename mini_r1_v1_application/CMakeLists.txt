cmake_minimum_required(VERSION 3.8)
project(mini_r1_v1_application)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rviz_common REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Core Widgets)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# RViz Panel Plugin Libraries
add_library(mission_control_panel SHARED
  src/mission_control_panel.cpp
  include/mini_r1_v1_application/mission_control_panel.hpp
)
target_include_directories(mission_control_panel PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>"
)
ament_target_dependencies(mission_control_panel
  rclcpp
  rviz_common
  visualization_msgs
  std_msgs
)
target_link_libraries(mission_control_panel Qt5::Widgets)

add_library(navigator_status_panel SHARED
  src/navigator_status_panel.cpp
  include/mini_r1_v1_application/navigator_status_panel.hpp
)
target_include_directories(navigator_status_panel PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>"
)
ament_target_dependencies(navigator_status_panel
  rclcpp
  rviz_common
  std_msgs
)
target_link_libraries(navigator_status_panel Qt5::Widgets)

# Install RViz Plugin Libraries
install(TARGETS mission_control_panel navigator_status_panel
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Export RViz Plugin XML
pluginlib_export_plugin_description_file(rviz_common rviz_plugins.xml)

# Install plugin description file
install(FILES
  rviz_plugins.xml
  DESTINATION share/${PROJECT_NAME}
)

# Install Python scripts
install(PROGRAMS
  scripts/marker_detector_node.py
  scripts/mission_zone_node.py
  scripts/dynamic_obstacle_node.py
  scripts/maze_navigator_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install nav_lib Python package alongside the scripts
install(DIRECTORY scripts/nav_lib/
  DESTINATION lib/${PROJECT_NAME}/nav_lib
)

# Install config files
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
