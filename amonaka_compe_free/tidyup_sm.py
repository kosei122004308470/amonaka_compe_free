#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""片付けタスクを実行する YASMIN ステートマシン."""

import rclpy
from navigation_tools.navlib import NavModule
from rclpy.node import Node
from tf2_ros import Buffer
from tf2_ros import TransformListener
from yasmin import Blackboard
from yasmin import StateMachine
from yasmin_viewer import YasminViewerPub

from carrobo_manipulation_pkg.hsrif import HSRInterfaces

from .states.bring_obj import BringObjState
from .states.grasp import GraspState
from .states.move2human import Move2HumanState
from .states.move2grasp import Move2GraspState
from .states.recog import RecogState
from .states.receive_order import ReceiveOrderState
from .task_context import TaskContext


class TidyupStateMachineNode(Node):
    """片付けタスクのステートマシンを構築して実行する ROS 2 ノード."""

    def __init__(self):
        """ロボットインターフェースとステートマシンを初期化する."""
        super().__init__('carrobo_tidyup')

        # carrobo_nav と carrobo_manipulation_pkg の例と同じインターフェースです。
        self.nav = NavModule()
        self.hsrif = HSRInterfaces()
        self.task_context = TaskContext()

        # Recog でカメラ座標系から base_link へ変換するために使います。
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.state_machine = StateMachine(outcomes=['SUCCEEDED', 'FAILED'])
        self.state_machine.add_state(
            name='Move2Human',
            state=Move2HumanState(self, self.nav, self.task_context),
            transitions={
                'succeeded': 'ReceiveOrder',
                'failed': 'FAILED',
            },
        )
        self.receive_order = ReceiveOrderState(
            self,
            self.hsrif,
            self.task_context,
        )
        self.state_machine.add_state(
            name='ReceiveOrder',
            state=self.receive_order,
            transitions={
                'succeeded': 'Move2Grasp',
                'failed': 'FAILED',
            },
        )
        self.state_machine.add_state(
            name='Move2Grasp',
            state=Move2GraspState(self, self.nav, self.task_context),
            transitions={
                'succeeded': 'Recog',
                'failed': 'FAILED',
            },
        )
        self.state_machine.add_state(
            name='Recog',
            state=RecogState(
                self,
                self.hsrif,
                self.tf_buffer,
                self.task_context,
            ),
            transitions={
                'succeeded': 'Grasp',
                'next_location': 'Move2Grasp',
                'failed': 'FAILED',
            },
        )
        self.state_machine.add_state(
            name='Grasp',
            state=GraspState(self, self.hsrif, self.task_context),
            transitions={
                'succeeded': 'BringObj',
                'failed': 'FAILED',
            },
        )
        self.state_machine.add_state(
            name='BringObj',
            state=BringObjState(self, self.nav, self.task_context),
            transitions={
                'succeeded': 'SUCCEEDED',
                'failed': 'FAILED',
            },
        )

        self.viewer = YasminViewerPub(
            fsm_name='CARROBO_TIDYUP',
            fsm=self.state_machine,
        )

        self.blackboard = Blackboard()

    def run(self) -> str:
        """ステートマシンを実行し、最終 outcome を返す."""
        outcome = self.state_machine(blackboard=self.blackboard)
        self.get_logger().info(f'State machine finished: {outcome}')
        return outcome

    def cleanup(self):
        """ナビゲーションと Viewer のバックグラウンド処理を止める."""
        self.receive_order.close()
        if self.nav.is_navigating:
            self.nav.cancel_nav_action()
        self.nav.shutdown()
        self.viewer.shutdown()


def main(args=None):
    """ROS 2 を初期化し、片付けステートマシンを実行する."""
    rclpy.init(args=args)
    node = TidyupStateMachineNode()

    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('片付けタスクを中断します。')
    finally:
        node.cleanup()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
