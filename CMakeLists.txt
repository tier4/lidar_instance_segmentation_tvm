cmake_minimum_required(VERSION 2.8.3)
project(lidar_instance_segmentation_tvm)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
  pcl_ros
  tf2
  tf2_geometry_msgs
  tf2_eigen
  tf2_ros
  autoware_perception_msgs
  dynamic_object_visualization
  cv_bridge
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_MWAITXINTRIN_H_INCLUDED")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STRICT_ANSI__")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O2 -fPIC")

catkin_package(
  INCLUDE_DIRS
    include
  CATKIN_DEPENDS
    roscpp
    roslib
    pcl_ros
    autoware_perception_msgs
    dynamic_object_visualization
)

include_directories(
  include
  lib/include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIR}
)

add_executable(lidar_instance_segmentation_tvm_node
  src/main.cpp
  src/node.cpp
  src/detector.cpp
  src/log_table.cpp
  src/feature_generator.cpp
  src/feature_map.cpp
  src/cluster2d.cpp
  src/debugger.cpp)

add_dependencies(lidar_instance_segmentation_tvm_node
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(lidar_instance_segmentation_tvm_node
  ${catkin_LIBRARIES}
  tvm_runtime
)

install(
  TARGETS
    lidar_instance_segmentation_tvm_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY data/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/data
)

install(DIRECTORY include/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}/${PROJECT_NAME}/
)
