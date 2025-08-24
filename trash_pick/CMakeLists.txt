cmake_minimum_required(VERSION 3.0.2)
project(trash_pick)

## C++11 예시 (필요시 사용)
# add_compile_options(-std=c++11)

## 필요한 catkin 패키지들
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  rospy
  tf2_geometry_msgs
  tf2_ros
  trajectory_msgs
  std_msgs
  message_generation
)

################################################
## Declare ROS messages, services and actions ##
################################################

## msg 파일 선언
add_message_files(
  FILES
  DetectedObject.msg
)

## msg 생성 (의존성 명시)
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  CATKIN_DEPENDS
    message_runtime
    std_msgs
    geometry_msgs
    tf2_geometry_msgs
    tf2_ros
    trajectory_msgs
)

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############

## 파이썬 스크립트 설치
catkin_install_python(
  PROGRAMS
    scripts/objdt.py
    scripts/ppap.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## 런치 파일 설치
install(FILES
  launch/trash_pick.launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

#############
## Testing ##
#############
# (필요시 추가)

