#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import rospy
import moveit_commander
import geometry_msgs.msg

def main():
    # MoveIt 초기화
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('free_move_and_return_task', anonymous=True)

    # 로봇팔 그룹을 제어하는 MoveGroupCommander 객체 생성
    arm_group = moveit_commander.MoveGroupCommander("arm")
    
    # 더 나은 계획을 위한 플래너 설정
    arm_group.set_planner_id("RRTConnectkConfigDefault")
    arm_group.set_planning_time(10)

    try:
        # 1. 안정적인 시작을 위해 'home' 자세로 먼저 이동
        rospy.loginfo("'home' 자세로 이동합니다...")
        arm_group.set_named_target("home")
        arm_group.go(wait=True)
        rospy.sleep(1)

        # 2. 자유 이동: 지정된 위치로 이동하기
        rospy.loginfo("지정된 위치로 자유롭게 이동합니다...")
        
        # 목표 위치 설정 (단위: 미터)
        target_position = [0.3, 0.0, 0.05] # [x, y, z]

	# 목표 Position으로 이동
        arm_group.set_position_target(target_position)
        arm_group.go(wait=True)
        
       

    except Exception as e:
        rospy.logerr(f"작업 중 에러 발생: {e}")
    finally:
        # MoveIt 종료
        moveit_commander.roscpp_shutdown()

if __name__ == '__main__':
    main()
