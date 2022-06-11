cmake_minimum_required(VERSION 3.06)

project(rm_detector)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)
find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        dynamic_reconfigure
        std_msgs
        cv_bridge
        nodelet
        pluginlib
        )

generate_dynamic_reconfigure_options(
        cfg/dynamic.cfg
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


find_package(OpenCV REQUIRED)

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
)


# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda-11.6/include)
link_directories(/usr/local/cuda-11.6/lib64)
# cudnn
include_directories(/opt/TensorRT-8.4.0.6/include)
link_directories(/opt/TensorRT-8.4.0.6/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

#add_executable(${PROJECT_NAME}_node src/detector_node.cpp src/detector.cpp)
##add_executable(${PROJECT_NAME}_node src/test.cpp)
#add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
#add_dependencies(${PROJECT_NAME}_node ${PROJECT_NAME}_gencfg)
#
#
#target_link_libraries(${PROJECT_NAME}_node nvinfer)
#target_link_libraries(${PROJECT_NAME}_node cudart)
#target_link_libraries(${PROJECT_NAME}_node
#        ${catkin_LIBRARIES}
#        ${OpenCV_LIBS}
#        )
#add_definitions(-O2 -pthread)

add_library(${PROJECT_NAME} src/detector.cpp)
add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_gencfg)


target_link_libraries(${PROJECT_NAME} nvinfer)
target_link_libraries(${PROJECT_NAME} cudart)
target_link_libraries(${PROJECT_NAME}
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        )
add_definitions(-O2 -pthread)
#