cmake_minimum_required(VERSION 3.8)
project(global_path_planning)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_srvs REQUIRED)
# Necessary for using GPMP2
find_package(Boost REQUIRED COMPONENTS regex timer serialization date_time chrono system filesystem program_options thread)
find_package(gpmp2 REQUIRED)
find_package(GTSAM REQUIRED)

# Debug messages
message(STATUS "GTSAM_INCLUDE_DIR: ${GTSAM_INCLUDE_DIR}")
message(STATUS "gpmp2_INCLUDE_DIR: ${gpmp2_INCLUDE_DIR}")
message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")

# Set compile definitions based on availability
if(GTSAM_FOUND AND gpmp2_FOUND)
  message(STATUS "GTSAM and GPMP2 found - full functionality enabled")
  add_compile_definitions(GPMP2_AVAILABLE)
else()
  if(NOT GTSAM_FOUND)
    message(WARNING "GTSAM not found!")
  endif()
  if(NOT gpmp2_FOUND)
    message(WARNING "GPMP2 not found!")
  endif()
endif()

# Create executable
add_executable(global_path_planner src/global_path_planner.cpp)

# Include directories
target_include_directories(global_path_planner PRIVATE
  ${GTSAM_INCLUDE_DIR}
  ${gpmp2_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(global_path_planner
  gpmp2
  gtsam
  ${Boost_LIBRARIES}
)

# ROS2 dependencies
ament_target_dependencies(global_path_planner
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  std_srvs
)

# Install launch files, config files, and rviz configs
install(DIRECTORY
  launch
  config
  rviz
  DESTINATION share/${PROJECT_NAME}/
)

# Target installation
install (TARGETS
  global_path_planner
  DESTINATION lib/${PROJECT_NAME}
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
