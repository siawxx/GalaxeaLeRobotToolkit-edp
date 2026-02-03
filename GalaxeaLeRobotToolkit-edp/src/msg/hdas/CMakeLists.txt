cmake_minimum_required(VERSION 3.16)
project(hdas_msg)

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

file(GLOB msg_files LIST_DIRECTORIES false RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "msg/*.msg")
file(GLOB srv_files LIST_DIRECTORIES false RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "srv/*.srv")

rosidl_generate_interfaces(${PROJECT_NAME}
        ${msg_files}
        ${srv_files}
        DEPENDENCIES std_msgs geometry_msgs sensor_msgs
)

ament_export_dependencies(rosidl_default_runtime std_msgs geometry_msgs sensor_msgs)

ament_package()
