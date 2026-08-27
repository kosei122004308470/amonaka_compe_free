#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""人から指示を受け取る位置まで移動するステート."""

import math

from geometry_msgs.msg import Pose2D
from navigation_tools.navlib import NavModule
from rclpy.node import Node
from yasmin import Blackboard
from yasmin import State

from ..task_context import TaskContext


# 人の位置 (x=0.8, y=2.0) の正面で、人の方を向く map 座標です。
GOAL_X = 2.4
GOAL_Y = 2.4
GOAL_YAW = math.pi

# 0.0 は到着するまで待ち続けます。必要なら秒数を指定してください。
NAVIGATION_TIMEOUT = 0.0


class Move2HumanState(State):
    """人から指示を受け取る固定位置へナビゲーションする."""

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
        """人の正面に設定した固定位置へ移動する."""
        self.node.get_logger().info('Executing state Move2Human')

        if GOAL_X is None or GOAL_Y is None or GOAL_YAW is None:
            self.node.get_logger().error(
                '人の前の位置が未設定です。move2human.py の '
                'GOAL_X / GOAL_Y / GOAL_YAW を設定してください。'
            )
            return 'failed'

        goal = Pose2D(
            x=float(GOAL_X),
            y=float(GOAL_Y),
            theta=float(GOAL_YAW),
        )
        self.node.get_logger().info(
            f'Navigation goal: x={goal.x:.2f}, y={goal.y:.2f}, '
            f'yaw={goal.theta:.2f}'
        )

        if self.nav.nav_goal(goal=goal, timeout=NAVIGATION_TIMEOUT):
            self.node.get_logger().info('人の前への移動が完了しました。')
            return 'succeeded'

        status = self.nav.nav_status
        message = getattr(status, 'message', str(status))
        self.node.get_logger().error(
            f'人の前への移動に失敗しました: {message}'
        )
        return 'failed'
