#!/usr/bin/env python3
import sys
import rospy
import moveit_commander

def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("gripper_test_direct", anonymous=True)

    gripper_group = moveit_commander.MoveGroupCommander("gripper")

    # 현재 조인트 값 가져오기
    joint_goal = gripper_group.get_current_joint_values()

    # 첫 번째 조인트(그리퍼 메인) 값 변경 (라디안)
    joint_goal[0] = 0.01   # open 방향
    joint_goal[1] = 0.0    # mimic joint

    rospy.loginfo("그리퍼 열기")
    gripper_group.set_joint_value_target(joint_goal)
    gripper_group.go(wait=True)
    rospy.sleep(2)

    rospy.loginfo("그리퍼 닫기")
    joint_goal[0] = -0.01  # close 방향
    joint_goal[1] = 0.0
    gripper_group.set_joint_value_target(joint_goal)
    gripper_group.go(wait=True)

    rospy.loginfo("그리퍼 테스트 완료")
    moveit_commander.roscpp_shutdown()

if __name__ == "__main__":
    main()

