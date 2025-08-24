#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import rospy
import moveit_commander
from geometry_msgs.msg import PointStamped
from trash_pick.msg import DetectedObject  # header, point(x,y,z), class_name
from std_msgs.msg import Bool

def clamp(v, lo, hi): return max(lo, min(hi, v))

class MoveToCan:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('move_to_can_tf', anonymous=True)

        # Arm / Gripper 그룹
        self.arm_group = moveit_commander.MoveGroupCommander("arm")
        self.gripper_group = moveit_commander.MoveGroupCommander("gripper")

        self.arm_group.set_planner_id("RRTConnectkConfigDefault")
        self.arm_group.set_planning_time(10)

        # ===== 좌표보정/대기 파라미터 =====
        self.x_offset = float(rospy.get_param('~x_offset', 0.030))
        self.y_scale  = float(rospy.get_param('~y_scale', 1.14))
        self.y_offset = float(rospy.get_param('~y_offset', 0.035))
        self.z_offset = float(rospy.get_param('~z_offset', -0.025))
        self.wait_before_move = float(rospy.get_param('~wait_before_move', 1.0))

        # ===== 그리퍼 한계 (prismatic) =====
        self.grip_open = float(rospy.get_param('~grip_open', -0.010))  # 완전 열림
        self.grip_min  = float(rospy.get_param('~grip_min',  -0.010))
        self.grip_max  = float(rospy.get_param('~grip_max',   0.019))

        # ===== 클래스별 “닫는 정도”와 “드롭 포즈” =====
        #  - paper/plastic/plastique 를 plastic 드롭포즈로 통일 처리
        self.cfgs = {
            'can': {
                'close_m': float(rospy.get_param('~aluminium_close_m', 0.002)),
                'drop_pose': rospy.get_param('~aluminium_drop_pose', 'can'),
            },
            'paper': {
                'close_m': float(rospy.get_param('~plastique_close_m', 0.001)),
                'drop_pose': rospy.get_param('~plastique_drop_pose', 'plastic'),
            },
            'plastic': {
                'close_m': float(rospy.get_param('~plastique_close_m', 0.001)),
                'drop_pose': rospy.get_param('~plastique_drop_pose', 'plastic'),
            },
            'plastique': {
                'close_m': float(rospy.get_param('~plastique_close_m', 0.001)),
                'drop_pose': rospy.get_param('~plastique_drop_pose', 'plastic'),
            },
        }

        # ===== 탐지 일시중지 컨트롤 =====
        self.pause_topic = rospy.get_param('~pause_topic', '/detect/pause')
        self.pause_pub = rospy.Publisher(self.pause_topic, Bool, queue_size=1, latch=True)
        self.pause_settle = float(rospy.get_param('~pause_settle', 0.15))   # pause 신호 안정화 대기
        self.resume_settle = float(rospy.get_param('~resume_settle', 0.15)) # resume 신호 안정화 대기
        self.busy = False  # 동작 중 콜백 무시

        # 구독(클래스+좌표)
        topic = rospy.get_param('~topic', '/can_target_point')
        rospy.Subscriber(topic, DetectedObject, self.callback)

        # === 초기 경로: home -> position3 -> position2 -> position1 ===
        self.busy = True
        try:
            self._pause_detection()
            rospy.loginfo("초기 경로 수행: home → position3 → position2 → position1")
            for n in ["position3", "position2", "position1"]:
                if not self.move_named(n):
                    rospy.logwarn(f"초기 경로 이동 실패: {n}")
                    break
        finally:
            self._resume_detection()
            self.busy = False

    # ---------- 유틸 ----------
    def _pause_detection(self):
        self.pause_pub.publish(Bool(data=True))
        rospy.sleep(self.pause_settle)

    def _resume_detection(self):
        self.pause_pub.publish(Bool(data=False))
        rospy.sleep(self.resume_settle)

    def set_gripper(self, pos_m, wait=True):
        pos_m = clamp(pos_m, self.grip_min, self.grip_max)
        joints = self.gripper_group.get_current_joint_values()
        joints[0] = pos_m
        self.gripper_group.set_joint_value_target(joints)
        self.gripper_group.go(wait=wait)

    def move_named(self, name):
        self.arm_group.set_named_target(name)
        ok = self.arm_group.go(wait=True)
        self.arm_group.stop()
        return ok

    def move_xyz(self, x, y, z):
        self.arm_group.set_position_target([x, y, z])
        ok = self.arm_group.go(wait=True)
        self.arm_group.stop()
        self.arm_group.clear_pose_targets()
        return ok

    # ---------- 메인 콜백 ----------
    def callback(self, msg):
        # 동작 중이면 들어온 좌표는 무시
        if self.busy:
            return

        self.busy = True
        try:
            # 외부 탐지 일시중지
            self._pause_detection()

            cls_raw = (msg.class_name or '').strip().lower()
            # 알 수 없는 클래스는 plastic 로 취급
            cls = cls_raw if cls_raw in self.cfgs else 'plastic'
            cfg = self.cfgs[cls]

            # 좌표 보정
            x = msg.point.x + self.x_offset
            y = (-msg.point.y + self.y_offset) * self.y_scale
            z = msg.point.z + self.z_offset

            rospy.loginfo(f"[{cls}] {self.wait_before_move:.1f}s 대기 후 position1→position2→position3→목표로 이동")
            rospy.sleep(self.wait_before_move)

            # === 요구사항 경로: position1 -> position2 -> position3 -> (목표 xyz) ===
            # 현재 position1에 있다고 가정하지만, 안전하게 강제 시퀀스 수행
            for n in ["position1", "position2", "position3"]:
                if not self.move_named(n):
                    rospy.logwarn(f"{n} 이동 실패 또는 계획 불가")
                    return

            # 목표 위치로 이동
            if not self.move_xyz(x, y, z):
                rospy.logwarn("목표 위치 이동 실패 또는 계획 불가")
                return

            # 클래스별 그리퍼 닫기 (집기)
            self.set_gripper(cfg['close_m'])
            rospy.sleep(0.3)

            # 드롭 포즈로 이동
            if not self.move_named(cfg['drop_pose']):
                rospy.logwarn(f"드롭 포즈 이동 실패: {cfg['drop_pose']}")
                return
            rospy.sleep(0.2)

            # 열어서 놓기
            self.set_gripper(self.grip_open)
            rospy.sleep(0.3)

            # === 복귀 경로: position3 → position2 → position1 === (home 대신)
            rospy.loginfo("복귀 경로 수행: position3 → position2 → position1")
            for n in ["position3", "position2", "position1"]:
                if not self.move_named(n):
                    rospy.logwarn(f"복귀 경로 이동 실패: {n}")
                    break

            rospy.loginfo("시퀀스 완료")

        finally:
            # 탐지 재개 + busy 해제는 항상 보장
            self._resume_detection()
            self.busy = False

if __name__ == '__main__':
    try:
        MoveToCan()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        moveit_commander.roscpp_shutdown()

