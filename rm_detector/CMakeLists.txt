cmake_minimum_required(VERSION 3.0.2)
project(rm_detector)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        dynamic_reconfigure
        std_msgs
        cv_bridge
        )
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(InferenceEngine REQUIRED)
find_package(ngraph REQUIRED)

generate_dynamic_reconfigure_options(
        cfg/dynamic.cfg
)

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
)


catkin_package(
        INCLUDE_DIRS
        include
        LIBRARIES
        CATKIN_DEPENDS
        roscpp
        dynamic_reconfigure
        std_msgs
        cv_bridge
        LIBRARIES ${PROJECT_NAME}
)

include_directories(
        include
        include/rm_detector
        ${catkin_INCLUDE_DIRS}
)
FILE(GLOB_RECURSE SOURCE_FILES "src/target/*.cpp")
## Declare a C++ library
add_library(${PROJECT_NAME} src/detector.cpp)

target_link_libraries(${PROJECT_NAME}
        ${catkin_LIBRARIES}
        ${InferenceEngine_LIBRARIES}
        ${NGRAPH_LIBRARIES}
        ${OpenCV_LIBS}
        )

add_executable(armor_detector_node src/detector_node.cpp src/detector.cpp)

add_dependencies(armor_detector_node ${PROJECT_NAME})
add_dependencies(armor_detector_node ${PROJECT_NAME}_gencfg)

target_link_libraries(
        armor_detector_node
        ${catkin_LIBRARIES}
        ${InferenceEngine_LIBRARIES}
        ${NGRAPH_LIBRARIES}
        ${OpenCV_LIBS}
)

