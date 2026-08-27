#!/usr/bin/env python3
# -*- encoding: UTF-8 -*-
#
# Copyright 2025 Hibikino-Musashi@Home
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Ryohei Kobayashi

import numpy as np
import rclpy
from geometry_msgs.msg import Pose2D
from navigation_tools.navlib import NavModule, NavStatus
from pumas_interfaces.action import PumasNav
from rclpy.node import Node

###############################
# 模範解答
#
# nav_goal() の引数を全部明示して呼ぶ．
#
# 普段は nav_goal(goal, timeout) だけで足りるが，
# それぞれの引数が mvn_pln の何を動かしているのかを一度確認しておく．
#
# nav_goal() は go_abs() の上位ラッパーで，中では
#   nav_goal() -> nav_goal_async() -> go_abs_async() -> /pumas_nav アクション
# と降りていく．送るアクションは go_abs() と同じ．
# 違うのは nav_goal_async() がゴールを送る前にやる前処理で，
#   ・腕と胴を走行姿勢へ戻す
#   ・障害物検知（点群）の on/off
#   ・走行中に一点を見続ける gaze
#   ・前のゴールの後始末（残った gaze のキャンセルなど）
# がここに入っている．これらが要らないなら go_abs() でよい．
###############################


class NavGoal(Node):
    """
    nav_goal() の全引数を指定して移動するノード．

    引数ごとの効果と，フィードバック・結果の読み方をまとめて確認します．
    """

    def __init__(self) -> None:
        """
        ノードの初期化処理を行うコンストラクタ．

        NavModule の生成を行います．
        """
        super().__init__('nav_goal')

        # NavModule は引数なしで生成する（内部で専用ノードとスピンスレッドを持つ）
        self.nav = NavModule()

    def on_feedback(self, status: NavStatus) -> None:
        """
        フィードバックのたびに呼ばれるコールバック．

        約 30 Hz で呼ばれる．重い処理や sleep を書いてはいけない．

        Args:
            status (NavStatus): 現在のナビゲーション状態のスナップショット．
        """

        # NavStatus を丸ごと見ることができる．ただし間引かずに出すと読みにくい
        # self.get_logger().info(str(status))
        pass

    def on_near_goal(self, feedback) -> None:
        """
        ゴール付近に入った瞬間に 1 回だけ呼ばれるコールバック．

        こちらは NavStatus ではなく PumasNav.Feedback が生で渡ってくる．
        1 ゴールにつき 1 回しか呼ばれない（立ち上がりのみ）．

        呼ばれない場合がある点に注意．mvn_pln が near_goal_reached を
        立てるのは SM_WAIT_FOR_MOVE_FINISHED の間に
        「残り < proximity_criterion（既定 0.2 m）」になった時だけなので，
        ゴールのすぐ近くから走り出すとその状態を素通りしてしまう．
        遠いゴールなら経路追従中に 0.2 m を切るので発火する．

        Args:
            feedback (PumasNav.Feedback): mvn_pln からのフィードバック．
        """
        # proximity_criterion はゴール付近と判定する半径．
        # mvn_pln 側の設定値がフィードバックに乗って届くので，
        # どの距離で発火したのかをここで確認できる．
        self.get_logger().warn(
            f'STATUS->NEAR_GOAL '
            f'(distance {feedback.remaining_distance:.2f} m, '
            f'proximity_criterion {feedback.proximity_criterion:.2f} m)'
        )
        self.get_logger().warn('you can start next some action')

    def run(self) -> None:
        """
        nav_goal() 絶対座標移動を全引数指定で呼び出すメソッド．
        """

        # -----------------------
        # argments
        goal = Pose2D(x=3.4, y=-1.2, theta=-1.57)  # [m], [rad]
        self.timeout = 0.0

        start_pose = {
            'arm_lift_joint': 0.0,
            'arm_flex_joint': np.deg2rad(0.0),
            'arm_roll_joint': np.deg2rad(0.0),
            'wrist_flex_joint': np.deg2rad(-90.0),
            'wrist_roll_joint': 0.0,
            'head_pan_joint': 0.0,
            'head_tilt_joint': np.deg2rad(0.0),
        }
        goal_pose = {
            'arm_lift_joint': 0.201,
            'arm_flex_joint': np.deg2rad(-118.0),
            'arm_roll_joint': np.deg2rad(5.0),
            'wrist_flex_joint': np.deg2rad(-62.0),
            'wrist_roll_joint': 0.0,
            'head_pan_joint': 0.0,
            'head_tilt_joint': np.deg2rad(-60.0),
        }

        motion_synth_pose = {
            'start': start_pose,
            'goal': goal_pose,
            'execution_time': 0.2,
        }
        motion_synth_pose = None

        goal_distance = 0.0   #<--ゴール手前で止まる距離
        use_point_cloud = True
        self.via_points = None
        omni_goal_yaw_align = True
        min_reach_goal_dist = 0.0
        # -----------------------

        self.get_logger().info(
            f'GOAL ({goal.x:.2f}, {goal.y:.2f}, {goal.theta:.2f})')

        # goalを送信する
        ok = self.nav.nav_goal(
            # ---------------------------------------------------------------
            # goal (Pose2D): 必須．map 座標系のゴール．
            #   x, y は [m]，theta は [rad]．
            # ---------------------------------------------------------------
            goal=goal,
            # ---------------------------------------------------------------
            # timeout (float): 必須．結果が確定するまで待つ上限 [s]．
            #   0（や None）なら確定するまで待ち続ける．
            #   時間切れになるとゴールをキャンセルして False を返し，
            #   nav_status.client_error が 'timeout' になる．
            # ---------------------------------------------------------------
            timeout=self.timeout,
            # ---------------------------------------------------------------
            # motion_synth_pose (dict | None): 移動しながら腕・胴を動かす設定．
            #   {'start': {関節名: 角度}, 'goal': {...}, 'execution_time': 秒}
            #   start は走り出しの姿勢，goal は到着時の姿勢．
            #   go_abs() では nav.motion_synth_start_pose などを自分で
            #   セットしてから motion_synth=True にする必要があったが，
            #   nav_goal() ではこの辞書 1 個で済む．
            #   None なら腕は走行姿勢（default_arm_pose）に戻される．
            # ---------------------------------------------------------------
            motion_synth_pose=motion_synth_pose,
            # ---------------------------------------------------------------
            # goal_distance (float | None): ゴールの手前で止める距離 [m]．
            #   pumas_nav2のmvn_pln(プランナー) 側でロボット -> ゴールの直線距離を見て停止する．
            #   止まったときは成功扱いで OUTCOME_STOP_SHORT が返る．
            #   0もしくはNone ならゴール直上まで移動．
            # ---------------------------------------------------------------
            goal_distance=goal_distance,
            # ---------------------------------------------------------------
            # use_point_cloud (bool): 点群由来の障害物を使うかどうか．
            #   ゴールを送る直前に navlib が param_rw 経由で
            #     potential_fields (斥力＝目の前の回避)
            #     map_augmenter   (コストマップ＝経路計画)
            #   の両方の use_point_cloud を同時に書き換える．
            #   LiDAR は use_lidar という別パラメータなので，False にしても
            #   完全に無防備にはならない．
            #
            #   注意 1: motion_synth_pose か gaze_point を使うと，
            #     この引数の値に関係なく強制的に False になる．
            #     腕を上げたり首を振ったりすると自分の腕がカメラに写り，
            #     それを障害物と誤検出して止まってしまうため．
            #   注意 2: 書き換えた値は走行後も残る．ただし nav_goal() は
            #     呼ぶたびにこの値を書き直すので，次に True で呼べば戻る．
            # ---------------------------------------------------------------
            use_point_cloud=use_point_cloud,
            # ---------------------------------------------------------------
            # gaze_point (str | bool): 走行中に見続ける TF の名前．
            #   False なら gaze を使わない．
            #   文字列を渡すとその TF を，True を渡すと gaze ノードの
            #   デフォルトの対象を見続ける．
            #   gaze 中は首を経路方向へ向ける move_head が切られ，
            #   走行が終わると navlib が自動で元へ戻す．
            # ---------------------------------------------------------------
            gaze_point=False,
            # ---------------------------------------------------------------
            # via_points (list[Pose2D] | None): 経路が必ず通る経由点のリスト．
            #   path_planner が「現在地 -> 経由点1 -> ... -> ゴール」を
            #   区間ごとに A* で解いてつなぐ．通したい通路があるときに使う．
            #   経由点は位置だけ見るので theta は経路に影響しない．
            #   0.5 m 以内まで近づくと「通過した」とみなして次の経由点へ進む．
            #   ゴールと違って経由点は代替地点への逃げがないので，1 つでも
            #   到達不能だと経路計画ごと失敗して OUTCOME_NO_PATH になる．
            #   None もしくは空リストなら普通の start -> goal 計画になる．
            # ---------------------------------------------------------------
            via_points=self.via_points,
            # ---------------------------------------------------------------
            # near_goal_callback (callable | None): ゴール付近に入った瞬間に
            #   1 回だけ呼ばれる．引数は PumasNav.Feedback．
            #   着く直前に手を伸ばし始めるような先読み処理に使う．
            #
            #   呼ばれない・気づかないときは以下を疑うこと．
            #   1. ログに埋もれている．
            #      1 ゴールにつき 1 回しか出ないので，feedback_callback を
            #      30 Hz で出しっぱなしにしていると流れて見えない．
            #   2. しきい値が近すぎて，そもそも発火していない．
            #      mvn_pln は SM_WAIT_FOR_MOVE_FINISHED の間に
            #      「ロボット -> ゴールの直線距離 < proximity_criterion」に
            #      なった時点で 1 回だけ near_goal_reached を立てる．
            #      この値は既定で 0.2 m しかないため（io/config の
            #      localization_params.yaml），経路追従がそれより手前で
            #      終わって最終角度補正へ抜けると一度も発火しない．
            #      先読みとして使いたいなら起動時に広げる:
            #        ros2 launch carrobo_slam navigation.launch.py \
            #             map_name:=carrobo proximity_criterion:=1.0
            # ---------------------------------------------------------------
            near_goal_callback=self.on_near_goal,
            # ---------------------------------------------------------------
            # feedback_callback (callable | None): フィードバックのたびに
            #   呼ばれる．引数は NavStatus．
            # ---------------------------------------------------------------
            feedback_callback=self.on_feedback,
            # ---------------------------------------------------------------
            # omni_goal_yaw_align (bool | None): 最後に全方位移動で
            #   ゴールの向きへ合わせるかどうか．次の動作にスムーズに移れる．
            # ---------------------------------------------------------------
            omni_goal_yaw_align=omni_goal_yaw_align,
            # ---------------------------------------------------------------
            # min_reach_goal_dist (float | None): 再計画を抑える半径 [m]．
            #   ゴールのこの距離以内から再計画しようとしたら，計画せずに停止する．
            # ---------------------------------------------------------------
            min_reach_goal_dist=min_reach_goal_dist,
        )

        # 移動結果を表示
        self.report(ok)

    def report(self, ok: bool) -> None:
        """
        移動結果を表示するメソッド．

        nav_goal() の戻り値は True / False だけなので，
        詳細は nav_status の outcome / message から読む．

        Args:
            ok (bool): nav_goal() が返した移動結果．
        """
        st = self.nav.nav_status

        # 走行中に mvn_pln が使っていた設定値．フィードバックに乗って届くので，
        # 「結局どの値が効いていたのか」をここで確認できる．
        self.get_logger().info(
            f'proximity_criterion={st.proximity_criterion:.2f} m, '
            f'goal_distance={st.goal_distance:.2f} m, '
            f'min_reach_goal_dist={st.min_reach_goal_dist:.2f} m'
        )

        if ok:
            self.get_logger().info(
                f'移動に成功しました (outcome={st.outcome_name}, {st.elapsed:.1f} 秒)'
            )
            # 成功にも種類がある．どういう着き方をしたのかは outcome で分かる．
            if st.outcome == PumasNav.Result.OUTCOME_GOAL_REACHED:
                self.get_logger().info('RESULT -> GOAL REACH')
            elif st.outcome == PumasNav.Result.OUTCOME_STOP_SHORT:
                self.get_logger().info(f'goal_distance の手前で停止 -> {st.message}')
            elif st.outcome == PumasNav.Result.OUTCOME_NEAR_GOAL_ACCEPTED:
                self.get_logger().info(f'ゴール付近を到達とみなした -> {st.message}')
            elif st.outcome == PumasNav.Result.OUTCOME_GOAL_RELAXED:
                self.get_logger().warn(f'ゴールの収束をあきらめて近い位置で停止 -> {st.message}')
            elif st.outcome == PumasNav.Result.OUTCOME_GOAL_RELOCATED:
                self.get_logger().warn(
                    f'ゴールが到達不能だったため近い位置で止まった -> {st.message}'
                )
            elif st.outcome == PumasNav.Result.OUTCOME_BLOCKED_STOPPED:
                self.get_logger().warn(f'障害物に囲まれて動けなくなった -> {st.message}')
            return

        # client_error が空でなければ mvn_pln まで届かなかった（or 待ちきれなかった）
        if st.client_error:
            self.get_logger().error(f'移動に失敗しました (クライアント側: {st.client_error})')
            if st.client_error == 'server_unavailable':
                self.get_logger().error(
                    'ナビゲーションシステムが起動しているか確認してください'
                    '（ros2 launch carrobo_slam navigation.launch.py map_name:=hogehoge）'
                )
            elif st.client_error == 'timeout':
                self.get_logger().error(
                    f'{self.timeout:.0f} 秒以内に着きませんでした．'
                    'timeout を延ばすか，再度送信してください'
                )
            return

        # ここまで来たら mvn_pln が中断を判断している
        self.get_logger().error(
            f'移動に失敗しました (outcome={st.outcome_name}) -> {st.message}')

        if st.outcome == PumasNav.Result.OUTCOME_NO_PATH:
            self.get_logger().error('ゴールまでの経路がありません．指定座標を見直してください')
            if self.via_points:
                self.get_logger().error('経由点が壁の中や到達不能な場所にないか確認してください')
        elif st.outcome == PumasNav.Result.OUTCOME_NOT_CONVERGING:
            self.get_logger().error('ゴールに近づけませんでした．周囲の障害物を確認してください')
        elif st.outcome == PumasNav.Result.OUTCOME_CANCELED:
            self.get_logger().error('ゴールがキャンセルされました')
        elif st.outcome == PumasNav.Result.OUTCOME_PREEMPTED:
            self.get_logger().error('別のゴールに上書きされました')


def main(args=None) -> None:
    """
    main 関数．

    ROS 2 クライアントライブラリの初期化，ノードの生成，
    全引数を指定した絶対移動を行います．
    また，Ctrl-C で終了するとノードを破棄してシャットダウンします．
    """
    # ROS 2 の Python クライアントライブラリの初期化
    rclpy.init(args=args)

    # NavGoal クラスのインスタンス生成
    node = NavGoal()

    try:
        # 全引数を指定して移動する
        node.run()

    except KeyboardInterrupt:
        node.nav.cancel_nav_action()

    finally:
        # NavModule は内部にノードをスピンするスレッドを持っているので止める
        node.nav.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
