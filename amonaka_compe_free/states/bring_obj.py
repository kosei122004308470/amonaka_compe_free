#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把持した物体を人の前まで運び、タスクを完了するステート."""

from geometry_msgs.msg import Pose2D
from navigation_tools.navlib import NavModule
from rclpy.node import Node
from yasmin import Blackboard
from yasmin import State

from ..task_context import TaskContext
from .move2human import GOAL_X
from .move2human import GOAL_Y
from .move2human import GOAL_YAW
from .move2human import NAVIGATION_TIMEOUT


class BringObjState(State):
    """把持した物体を人の正面まで運ぶ."""

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
        """人の前へ移動し、運んだ物体名を表示する."""
        self.node.get_logger().info('Executing state BringObj')

        if self.context.recognized_object is None:
            self.node.get_logger().error(
                'TaskContext に運ぶ物体名がありません。'
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

        if not self.nav.nav_goal(goal=goal, timeout=NAVIGATION_TIMEOUT):
            status = self.nav.nav_status
            message = getattr(status, 'message', str(status))
            self.node.get_logger().error(
                f'物体を人の前へ運べませんでした: {message}'
            )
            return 'failed'

        self.node.get_logger().info('人の前への移動が完了しました。')
        self.node.get_logger().info(
            f'選んだ物体: {self.context.recognized_object}'
        )
        return 'succeeded'
