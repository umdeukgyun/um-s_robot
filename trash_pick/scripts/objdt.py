#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import rospy
import torch
import cv2
import numpy as np
import pyrealsense2 as rs
from geometry_msgs.msg import PointStamped
import tf2_ros
import tf2_geometry_msgs
from math import sqrt
from std_msgs.msg import Bool

from trash_pick.msg import DetectedObject  # 커스텀 메시지

def robust_depth_from_roi(depth_frame, cx, cy, roi_half, depth_scale):
    depth_img = np.asanyarray(depth_frame.get_data())  # uint16
    h, w = depth_img.shape
    x1 = max(0, cx - roi_half); x2 = min(w, cx + roi_half + 1)
    y1 = max(0, cy - roi_half); y2 = min(h, cy + roi_half + 1)
    roi = depth_img[y1:y2, x1:x2].astype(np.float32) * depth_scale  # meters
    vals = roi.reshape(-1); vals = vals[vals > 0]
    if vals.size < 9: return None
    lo, hi = np.percentile(vals, [10, 90])
    vals = vals[(vals >= lo) & (vals <= hi)]
    if vals.size == 0: return None
    return float(np.median(vals))

def parse_classes_param(s):
    if isinstance(s, list): return [x.strip() for x in s]
    return [x.strip() for x in str(s).split(",") if x.strip()]

def dist3(p, q):
    return sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2 + (p[2]-q[2])**2)

def main():
    rospy.init_node('objdt', anonymous=True)

    # === 파라미터 ===
    model_path      = rospy.get_param('~model_path', os.path.expanduser('~/yolov5/runs/train/hope5/weights/best.pt'))
    conf_thresh     = float(rospy.get_param('~conf', 0.5))
    cooldown        = float(rospy.get_param('~cooldown', 3.0))
    z_min           = float(rospy.get_param('~z_min', -0.15))
    z_max           = float(rospy.get_param('~z_max', 1.50))
    roi_half        = int(rospy.get_param('~roi_half', 5))
    allowed_classes = parse_classes_param(rospy.get_param('~allowed_classes', 'can,paper'))
    frame_camera    = rospy.get_param('~camera_frame', 'camera_link')
    frame_target    = rospy.get_param('~target_frame', 'link1')  # TF 변환 목적 프레임

    # 안정화/판정 파라미터
    position_tolerance = float(rospy.get_param('~position_tolerance', 0.01))  # m
    stable_required    = float(rospy.get_param('~stable_required', 1.0))      # s

    # 객체당 1회 발송 파라미터
    new_obj_dist   = float(rospy.get_param('~new_obj_dist', 0.05))   # m
    forget_after   = float(rospy.get_param('~forget_after', 300.0))  # s
    max_memory     = int(rospy.get_param('~max_memory', 100))

    # --- 추가 파라미터 ---
    repeat_block_s = float(rospy.get_param('~repeat_block_s', 5.0))      # 이 시간 지나면 같은 자리/클래스도 재발행 허용
    pause_topic    = rospy.get_param('~pause_topic', '/detect/pause')     # 제어노드가 퍼블리시하는 일시정지 토픽

    # === 상태 ===
    pick_busy = False                # 로봇팔 동작 중이면 True
    _prev_busy = False               # busy 엣지 감지용
    last_pub_time = 0.0
    candidate_pos = None
    stable_since  = None
    published = []                   # [{'pos':(x,y,z), 'cls':'can', 't':time}, ...]

    # === 콜백 ===
    def busy_cb(msg):
        nonlocal pick_busy, candidate_pos, stable_since, published, _prev_busy
        pick_busy = bool(msg.data)

        # rising edge: idle -> busy
        if pick_busy and not _prev_busy:
            # 진행 중 후보/타이머 리셋해서 중간 발송 방지
            candidate_pos = None
            stable_since  = None
            rospy.loginfo("[objdt] pick_busy=True → 감지/발송 일시 중단")

        # falling edge: busy -> idle
        if (not pick_busy) and _prev_busy:
            # 같은 클래스 연속 발행을 막던 메모리 비움 (새 사이클 허용)
            published.clear()
            rospy.loginfo("[objdt] pick_busy=False → 감지/발송 재개 (중복 메모리 리셋)")

        _prev_busy = pick_busy

    def prune_published(now):
        nonlocal published
        published = [o for o in published if now - o['t'] <= forget_after]
        if len(published) > max_memory:
            published = published[-max_memory:]

    def is_duplicate(cur_pos, cls, now):
        # repeat_block_s 이내 + 거리 new_obj_dist 이내 + 같은 클래스면 '중복'
        prune_published(now)
        for o in published:
            if (now - o['t'] <= repeat_block_s) and (o['cls'] == cls) and (dist3(cur_pos, o['pos']) < new_obj_dist):
                return True
        return False

    # === YOLO 로드 ===
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    model.conf = conf_thresh

    # === RealSense ===
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    # === ROS Pub/TF & Sub ===
    pub = rospy.Publisher('/can_target_point', DetectedObject, queue_size=10)
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    rospy.Subscriber(pause_topic, Bool, busy_cb, queue_size=1)

    try:
        while not rospy.is_shutdown():
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())

            # 로봇팔이 바쁜 동안엔 감지/발송 중단 (뷰만 표시)
            if pick_busy:
                view = color_img.copy()
                cv2.putText(view, "BUSY: waiting arm...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("YOLO View", view)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # === YOLO 추론 ===
            results = model(color_img)
            det = results.xyxy[0]  # (x1,y1,x2,y2, conf, cls)

            selected = None
            if det is not None and len(det) > 0:
                rows = det.cpu().numpy()
                rows = sorted(rows, key=lambda r: r[4], reverse=True)
                for x1, y1, x2, y2, conf, cls_idx in rows:
                    class_name = model.names[int(cls_idx)]
                    if class_name in allowed_classes and conf >= conf_thresh:
                        selected = (int(x1), int(y1), int(x2), int(y2), float(conf), class_name)
                        break

            if selected is None:
                candidate_pos = None; stable_since = None
                rospy.loginfo_throttle(5.0, "[objdt] 유효 탐지 없음")
                cv2.imshow("YOLO View", results.render()[0])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            x1, y1, x2, y2, conf, class_name = selected
            now = time.time()

            # 전역 쿨다운
            if now - last_pub_time < cooldown:
                cv2.imshow("YOLO View", results.render()[0])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            cx = (x1 + x2) // 2; cy = (y1 + y2) // 2

            # 깊이(m) 추출
            depth_m = robust_depth_from_roi(depth_frame, cx, cy, roi_half, depth_scale)
            if depth_m is None or np.isnan(depth_m) or depth_m <= 0:
                rospy.logwarn("[objdt] ROI 깊이 추출 실패")
                cv2.imshow("YOLO View", results.render()[0])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # 픽셀 → 카메라 좌표
            intr = color_frame.profile.as_video_stream_profile().intrinsics
            X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], depth_m)

            # camera_link 기준 포인트
            pt = PointStamped()
            pt.header.stamp = rospy.Time.now()
            pt.header.frame_id = frame_camera
            pt.point.x, pt.point.y, pt.point.z = X, Y, Z

            try:
                tfm = tf_buffer.lookup_transform(frame_target, frame_camera, rospy.Time(0), rospy.Duration(1.0))
                tpt = tf2_geometry_msgs.do_transform_point(pt, tfm)

                # z 범위 필터
                if tpt.point.z < z_min or tpt.point.z > z_max:
                    rospy.logwarn(f"[objdt] z 범위 벗어남: {tpt.point.z:.3f} m")
                    cv2.imshow("YOLO View", results.render()[0])
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                # 안정화 체크
                cur_pos = (tpt.point.x, tpt.point.y, tpt.point.z)

                def within_tol(p, q, tol):
                    return abs(p[0]-q[0]) < tol and abs(p[1]-q[1]) < tol and abs(p[2]-q[2]) < tol

                if candidate_pos is None or not within_tol(cur_pos, candidate_pos, position_tolerance):
                    candidate_pos = cur_pos; stable_since = now
                    rospy.loginfo(f"[objdt] 새 후보 등록, 안정화 대기: {cur_pos}")
                else:
                    elapsed = now - (stable_since or now)
                    if elapsed >= stable_required:
                        # === 객체당 1회 발송: 중복 체크 (시간 조건 포함) ===
                        if is_duplicate(cur_pos, class_name, now):
                            rospy.loginfo_throttle(1.0, "[objdt] 동일 클래스/근접 위치(시간 내) → 발송 생략")
                        else:
                            # 퍼블리시
                            msg = DetectedObject()
                            msg.header.stamp = rospy.Time.now()
                            msg.header.frame_id = frame_target
                            msg.point.x, msg.point.y, msg.point.z = cur_pos
                            msg.class_name = class_name
                            pub.publish(msg)

                            last_pub_time = now
                            published.append({'pos': cur_pos, 'cls': class_name, 't': now})
                            prune_published(now)

                            rospy.loginfo(f"[objdt] NEW OBJ → PUB {class_name} @ {frame_target}: "
                                          f"({cur_pos[0]:.3f}, {cur_pos[1]:.3f}, {cur_pos[2]:.3f})")

                            # 퍼블리시 직후 초기화 (곧 busy=True 들어올 것)
                            candidate_pos = None
                            stable_since  = None
                    else:
                        rospy.loginfo_throttle(1.0, f"[objdt] 안정화 진행중 {elapsed:.2f}/{stable_required:.2f}s")

            except Exception as e:
                rospy.logwarn(f"[objdt] TF 변환 실패: {e}")

            # 뷰
            view = results.render()[0].copy()
            cv2.circle(view, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(view, f"{class_name} {conf:.2f}  {depth_m:.2f}m", (cx, max(0, cy - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.imshow("YOLO View", view)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        try: pipeline.stop()
        except Exception: pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

