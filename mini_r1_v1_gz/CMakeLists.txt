cmake_minimum_required(VERSION 3.8)
project(mini_r1_v1_gz)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ros_gz_sim REQUIRED)

find_package(Qt5 COMPONENTS Core Widgets Gui REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rviz_common REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(pluginlib REQUIRED)

set(CMAKE_AUTOMOC ON)

add_library(rviz_teleop_plugin SHARED
  src/teleop_panel.cpp
  include/mini_r1_v1_gz/teleop_panel.hpp
)

target_include_directories(rviz_teleop_plugin
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(rviz_teleop_plugin
  Qt5::Widgets
)

ament_target_dependencies(rviz_teleop_plugin
  rclcpp
  rviz_common
  geometry_msgs
  pluginlib
)

install(TARGETS rviz_teleop_plugin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

install(FILES plugin_description.xml
  DESTINATION share/${PROJECT_NAME}
)

pluginlib_export_plugin_description_file(rviz_common plugin_description.xml)

ament_export_libraries(rviz_teleop_plugin)

install( DIRECTORY launch config rviz
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
