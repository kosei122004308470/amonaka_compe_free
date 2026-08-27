#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPT が選んだ把持場所まで移動するステート."""

import math

from geometry_msgs.msg import Pose2D
from navigation_tools.navlib import NavModule
from rclpy.node import Node
from yasmin import Blackboard
from yasmin import State

from ..task_context import TaskContext


# 0.0 は到着するまで待ち続けます。必要なら秒数を指定してください。
NAVIGATION_TIMEOUT = 0.0


class Move2GraspState(State):
    """TaskContext の次候補へナビゲーションする."""

    def __init__(
        self,
        node: Node,
        nav: NavModule,
        context: TaskContext,
    ):
        """ステートを初期化する."""
        super().__init__(outcomes=['succeeded', 'failed'])
        self.node = node
        self.nav = nav
        self.context = context

    def execute(self, _: Blackboard) -> str:
        """キューから次の把持場所を取り出して移動する."""
        self.node.get_logger().info('Executing state Move2Grasp')

        target = self.context.pop_grasp_goal()
        if target is None:
            self.node.get_logger().error('移動可能な把持場所候補がありません。')
            return 'failed'

        try:
            coordinates = tuple(
                float(value) for value in (target.x, target.y, target.yaw)
            )
        except (TypeError, ValueError):
            self.node.get_logger().error(
                f'把持場所 {target.name} の座標を数値に変換できません。'
            )
            return 'failed'

        if not all(math.isfinite(value) for value in coordinates):
            self.node.get_logger().error(
                f'把持場所 {target.name} の座標が不正です。'
            )
            return 'failed'

        x, y, yaw = coordinates
        goal = Pose2D(
            x=x,
            y=y,
            theta=yaw,
        )
        self.node.get_logger().info(
            f'Navigation target: {target.name}, '
            f'x={goal.x:.2f}, y={goal.y:.2f}, yaw={goal.theta:.2f}'
        )

        if self.nav.nav_goal(goal=goal, timeout=NAVIGATION_TIMEOUT):
            self.node.get_logger().info(
                f'把持場所 {target.name} への移動が完了しました。'
                f'残り候補数: {len(self.context.grasp_goals)}'
            )
            return 'succeeded'

        status = self.nav.nav_status
        message = getattr(status, 'message', str(status))
        self.node.get_logger().error(
            f'把持場所 {target.name} への移動に失敗しました: {message}'
        )
        return 'failed'
