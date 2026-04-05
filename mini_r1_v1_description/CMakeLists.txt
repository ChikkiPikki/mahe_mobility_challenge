cmake_minimum_required(VERSION 3.8)
project(mini_r1_v1_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(ros_gz_bridge REQUIRED)
find_package(ros_gz_sim REQUIRED)

add_custom_target(generate_world ALL
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/worlds/download_models.py ${CMAKE_CURRENT_SOURCE_DIR}/worlds/multi_floor_college.building.yaml
  COMMAND bash -c "source ${CMAKE_INSTALL_PREFIX}/../../install/setup.bash && ros2 run rmf_building_map_tools building_map_generator gazebo ${CMAKE_CURRENT_SOURCE_DIR}/worlds/multi_floor_college.building.yaml ${CMAKE_CURRENT_SOURCE_DIR}/worlds/output.world ${CMAKE_CURRENT_SOURCE_DIR}/worlds/models"
  COMMAND bash -c "if ! grep -q 'gz-sim-sensors-system' ${CMAKE_CURRENT_SOURCE_DIR}/worlds/output.world; then sed -i '/<plugin filename=\"libgz-sim-physics-system.so\"/i \\    <plugin filename=\"libgz-sim-sensors-system.so\" name=\"gz::sim::systems::Sensors\">\\n      <render_engine>ogre2</render_engine>\\n    </plugin>' ${CMAKE_CURRENT_SOURCE_DIR}/worlds/output.world; fi"
  COMMAND bash -c "if ! grep -q 'gz-sim-imu-system' ${CMAKE_CURRENT_SOURCE_DIR}/worlds/output.world; then sed -i '/<plugin filename=\"libgz-sim-physics-system.so\"/i \\    <plugin filename=\"libgz-sim-imu-system.so\" name=\"gz::sim::systems::Imu\">\\n    </plugin>' ${CMAKE_CURRENT_SOURCE_DIR}/worlds/output.world; fi"
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/worlds/generate_markers_and_signs.py ${CMAKE_CURRENT_SOURCE_DIR}/worlds/multi_floor_college.building.yaml ${CMAKE_CURRENT_SOURCE_DIR}/worlds/output.world ${CMAKE_CURRENT_SOURCE_DIR}/worlds
  COMMENT "Generating Gazebo world from RMF building map..."
  VERBATIM
)

install(
  DIRECTORY config launch meshes urdf worlds
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
