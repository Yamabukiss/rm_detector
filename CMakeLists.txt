cmake_minimum_required(VERSION 3.0.2)
project(rm_detector)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        dynamic_reconfigure
        std_msgs
        cv_bridge
        nodelet
        pluginlib
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
        ${catkin_INCLUDE_DIRS}
)
## Declare a C++ library
add_library(${PROJECT_NAME} src/detector.cpp)

target_link_libraries(${PROJECT_NAME}
        ${catkin_LIBRARIES}
        ${InferenceEngine_LIBRARIES}
        ${NGRAPH_LIBRARIES}
        ${OpenCV_LIBS}
        )


add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_gencfg)

target_link_libraries(
        ${PROJECT_NAME}
        ${catkin_LIBRARIES}
        ${InferenceEngine_LIBRARIES}
        ${NGRAPH_LIBRARIES}
        ${OpenCV_LIBS}
)