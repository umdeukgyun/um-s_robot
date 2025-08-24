#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
from geometry_msgs.msg import PointStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class MoveToCan:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('move_to_can_tf', anonymous=True)

        self.arm_group = moveit_commander.MoveGroupCommander("arm")
        self.arm_group.set_planner_id("RRTConnectkConfigDefault")
        self.arm_group.set_planning_time(10)

        # 그리퍼 제어 퍼블리셔 (두 조인트 제어용)
        self.gripper_pub = rospy.Publisher('/gripper/command', JointTrajectory, queue_size=10)

        rospy.Subscriber('/can_target_point', PointStamped, self.callback)

        rospy.loginfo("로봇팔 'home' 자세로 이동 중...")
        self.arm_group.set_named_target("home")
        self.arm_group.go(wait=True)

    def callback(self, msg):
        x = msg.point.x +0.05
        y = -msg.point.y -0.027
        z = msg.point.z +0.02  # 살짝 위에서 접근

        rospy.loginfo(f"5초 대기 후 이동 예정 → x:{x:.3f} y:{y:.3f} z:{z:.3f}")
        rospy.sleep(5.0)

        self.arm_group.set_position_target([x, y, z])
        plan_success = self.arm_group.go(wait=True)

        if plan_success:
            rospy.loginfo("이동 성공! 그리퍼 동작 후 home 복귀")
            rospy.sleep(1.0)

            # 그리퍼 반쯤 닫기
            traj = JointTrajectory()
            traj.joint_names = ['gripper', 'gripper_sub']  # 조인트 이름 반드시 확인!
            point = JointTrajectoryPoint()
            point.positions = [0.006, 0.006]
            point.time_from_start = rospy.Duration(1.0)
            traj.points.append(point)
            self.gripper_pub.publish(traj)

            rospy.sleep(1.0)

            self.arm_group.set_named_target("home")
            self.arm_group.go(wait=True)
        else:
            rospy.logwarn("이동 실패 또는 계획 불가")

if __name__ == '__main__':
    try:
        MoveToCan()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        moveit_commander.roscpp_shutdown()

