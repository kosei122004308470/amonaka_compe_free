#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLOv8 の検出結果から把持姿勢を求めるステート."""

import math
import time

import rclpy
import tf2_geometry_msgs  # noqa: F401
import tf_transformations as tft
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Quaternion
from grasp_point_detection_interfaces.srv import GraspPointService
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_ros import Buffer
from tf2_ros import TransformException
from yasmin import Blackboard
from yasmin import State
from yolov8_detection_interfaces.srv import ObjectDetectionService

from carrobo_manipulation_pkg.hsrif import HSRInterfaces

from ..task_context import TaskContext


CONFIDENCE_THRESHOLD = 0.6
MAX_GRASP_DISTANCE = 2.0
TALL_THRESHOLD = 0.15


class RecogState(State):
    """物体検出と把持点推定を行うステート."""

    def __init__(
        self,
        node: Node,
        hsrif: HSRInterfaces,
        tf_buffer: Buffer,
        context: TaskContext,
    ):
        """サービスクライアントを生成する."""
        super().__init__(outcomes=['succeeded', 'next_location', 'failed'])
        self.node = node
        self.hsrif = hsrif
        self.tf_buffer = tf_buffer
        self.context = context

        self.detect_client = self.node.create_client(
            ObjectDetectionService, '/yolov8_detection/service'
        )
        self.grasp_client = self.node.create_client(
            GraspPointService, '/grasp_point_detection/service'
        )

    def _wait_for_service(self, client, service_name: str) -> bool:
        """サービスが利用可能になるまで待つ."""
        while rclpy.ok():
            if client.wait_for_service(timeout_sec=1.0):
                return True
            self.node.get_logger().info(
                f'{service_name} が見つかりません。起動を待っています...'
            )
        return False

    @staticmethod
    def _select_target(detections, target_names) -> int:
        """対象候補に一致する検出のうち最高スコアの添字を返す."""
        expected_names = set(target_names)
        candidates = [
            (index, bbox.score)
            for index, bbox in enumerate(detections.bbox)
            if bbox.name in expected_names
        ]
        if not candidates:
            return -1
        return max(candidates, key=lambda item: item[1])[0]

    def _not_found_outcome(self) -> str:
        """探索候補の残数に応じて次地点または失敗を返す."""
        self.context.clear_grasp_result()
        targets = ', '.join(self.context.target_objects) or '対象物体'
        if self.context.has_pending_grasp_goals:
            self.node.get_logger().warning(
                f'{targets} が見つかりません。次の探索場所へ移動します。'
            )
            return 'next_location'
        self.node.get_logger().error(
            f'{targets} が見つからず、探索場所候補も残っていません。'
        )
        return 'failed'

    def _move_to_recognition_pose(self) -> bool:
        """現在地に設定された認識用関節姿勢へ移動する."""
        goal = self.context.current_grasp_goal
        if goal is None:
            self.node.get_logger().error('現在の探索場所が設定されていません。')
            return False

        joint_positions = goal.recognition_joint_positions()
        if not joint_positions:
            self.node.get_logger().error(
                f'探索場所 {goal.name} の認識用 joint 設定がありません。'
            )
            return False
        try:
            self.hsrif.whole_body.move_to_joint_positions(
                joint_positions,
                sync=True,
            )
        except Exception as error:
            self.node.get_logger().error(
                f'探索場所 {goal.name} の認識姿勢へ移動できません: {error}'
            )
            return False
        self.node.get_logger().info(
            f'{goal.name} の認識用 joint 設定を適用しました。'
        )
        return True

    def execute(self, _: Blackboard) -> str:
        """対象物体を認識し、把持前姿勢を TaskContext に保存する."""
        self.node.get_logger().info('Executing state Recog')
        self.context.clear_grasp_result()

        if not self.context.target_objects:
            self.node.get_logger().error('探索対象物体が設定されていません。')
            return 'failed'

        if not self._wait_for_service(
            self.detect_client, '/yolov8_detection/service'
        ):
            return 'failed'
        if not self._wait_for_service(
            self.grasp_client, '/grasp_point_detection/service'
        ):
            return 'failed'

        # carrobo_move.yaml の各場所に定義したカメラ・アーム姿勢を使う。
        if not self._move_to_recognition_pose():
            return 'failed'
        time.sleep(2.0)

        # TF を受信して Buffer に溜めてからサービスを呼びます。
        for _ in range(20):
            rclpy.spin_once(self.node, timeout_sec=0.1)

        detect_req = ObjectDetectionService.Request()
        detect_req.confidence_th = CONFIDENCE_THRESHOLD
        future = self.detect_client.call_async(detect_req)
        rclpy.spin_until_future_complete(self.node, future)
        response = future.result()

        if response is None:
            self.node.get_logger().error(
                '物体検出サービスから応答がありません。'
            )
            return 'failed'

        detections = response.detections
        names = [bbox.name for bbox in detections.bbox]
        self.node.get_logger().info(f'検出した物体: {names}')

        index = self._select_target(
            detections,
            self.context.target_objects,
        )
        if index < 0:
            return self._not_found_outcome()
        if index >= len(detections.segments):
            self.node.get_logger().error(
                '検出物体に対応するマスクがありません。'
            )
            return 'failed'

        grasp_req = GraspPointService.Request()
        grasp_req.depth = detections.depth
        grasp_req.mask = detections.segments[index]
        grasp_req.camera_info = detections.camera_info
        grasp_req.max_distance = MAX_GRASP_DISTANCE

        future = self.grasp_client.call_async(grasp_req)
        rclpy.spin_until_future_complete(self.node, future)
        grasp_response = future.result()

        if grasp_response is None:
            self.node.get_logger().error(
                '把持点推定サービスから応答がありません。'
            )
            return 'failed'
        if not grasp_response.success:
            self.node.get_logger().error(
                f'把持点を推定できませんでした: {grasp_response.message}'
            )
            return 'failed'

        # 推定結果を、腕を動かす base_link 基準へ変換します。
        stamped = PoseStamped()
        stamped.header = detections.camera_info.header
        stamped.pose = grasp_response.grasp.pose
        try:
            object_pose = self.tf_buffer.transform(
                stamped,
                'base_link',
                timeout=Duration(seconds=2.0),
            ).pose
        except TransformException as error:
            self.node.get_logger().error(
                f'把持姿勢の TF 変換に失敗しました: {error}'
            )
            return 'failed'

        # carrobo_manipulation_pkg の把持例と同じルールで掴む向きを決めます。
        height = grasp_response.grasp.size.z
        if height > TALL_THRESHOLD:
            self.node.get_logger().info('背が高い物体なので横から掴みます。')
            roll = math.pi
            pitch = -math.pi / 2.0
            object_pose.position.x -= 0.1
            approach = 0.1
        else:
            self.node.get_logger().info('平たい物体なので上から掴みます。')
            roll = math.pi
            pitch = 0.0
            object_pose.position.z += 0.15
            approach = 0.09

        qx, qy, qz, qw = tft.quaternion_from_euler(roll, pitch, 0.0)
        object_pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)

        target_name = detections.bbox[index].name
        self.context.set_grasp_result(
            target_name,
            object_pose,
            approach,
        )
        self.node.get_logger().info(
            f'{target_name} の把持姿勢を TaskContext に保存しました。'
        )
        return 'succeeded'
