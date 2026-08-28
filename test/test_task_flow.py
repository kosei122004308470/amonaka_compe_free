"""借り物競争ステート間の分岐とデータ受け渡しを検証する."""

import math
from types import SimpleNamespace

from amonaka_compe_free.states.bring_obj import BringObjState
from amonaka_compe_free.states.grasp import GRASP_FORCE
from amonaka_compe_free.states.grasp import GraspState
from amonaka_compe_free.states.recog import RecogState
from amonaka_compe_free.states.receive_order import ReceiveOrderState
from amonaka_compe_free.prediction_result_converter import (
    PredictionResultConverter,
)
from amonaka_compe_free.task_context import NavigationGoal
from amonaka_compe_free.task_context import TaskContext


class _Logger:
    """テスト中のINFOログを保存する."""

    def __init__(self):
        """ログ保存先を初期化する."""
        self.messages = []

    def info(self, message):
        """INFOログを保存する."""
        self.messages.append(message)

    def warning(self, _):
        """WARNINGログを破棄する."""

    def error(self, _):
        """ERRORログを破棄する."""


def test_task_context_pops_navigation_goals_in_fifo_order():
    """把持場所候補を登録順に取り出す."""
    first = NavigationGoal('first', 1.0, 2.0, 0.0)
    second = NavigationGoal('second', 3.0, 4.0, 1.0)
    context = TaskContext()

    context.begin_order(['cup'], [first, second], {'target_object': 'cup'})

    assert context.pop_grasp_goal() is first
    assert context.has_pending_grasp_goals
    assert context.pop_grasp_goal() is second
    assert not context.has_pending_grasp_goals
    assert context.pop_grasp_goal() is None


def test_navigation_goal_keeps_recognition_joint_positions():
    """場所ごとの認識姿勢をナビゲーション目標と一緒に渡す."""
    goal = NavigationGoal(
        'table',
        1.0,
        2.0,
        0.0,
        (('head_tilt_joint', math.radians(-40.0)), ('arm_lift_joint', 0.5)),
    )

    assert goal.recognition_joint_positions() == {
        'head_tilt_joint': math.radians(-40.0),
        'arm_lift_joint': 0.5,
    }


def test_receive_order_parses_map_and_joint_sections():
    """新形式のcarrobo_move設定を移動・認識目標へ変換する."""
    context = TaskContext()
    state = SimpleNamespace(
        context=context,
        _leaf_names=ReceiveOrderState._leaf_names,
        _load_inputs=lambda: (
            {'food': {'apple': None}},
            {
                'pick_table': {
                    'map': {'x': 1.5, 'y': -2.0, 'theta': 90.0},
                    'joint': {
                        'head_tilt_joint': -40.0,
                        'arm_lift_joint': 0.5,
                    },
                }
            },
            {},
        ),
    )
    result = {
        'target_object': 'apple',
        'target_location': 'pick_table',
        'confidence': 0.9,
        'pointing_direction': 'east',
        'object_direction': 'north',
        'reason': 'test',
        'search_priority': [],
    }

    ReceiveOrderState._validate_and_update(state, result)

    goal = context.pop_grasp_goal()
    assert goal is not None
    assert (goal.x, goal.y, goal.yaw) == (1.5, -2.0, math.pi / 2.0)
    assert goal.recognition_joint_positions() == {
        'head_tilt_joint': math.radians(-40.0),
        'arm_lift_joint': 0.5,
    }


def test_recog_selects_highest_scoring_expected_object():
    """対象候補以外を除外して最高スコアの検出を選ぶ."""
    detections = SimpleNamespace(
        bbox=[
            SimpleNamespace(name='robot', score=0.99),
            SimpleNamespace(name='cup', score=0.80),
            SimpleNamespace(name='mug', score=0.90),
        ]
    )

    index = RecogState._select_target(detections, ['cup', 'mug'])

    assert index == 2


def test_recog_moves_to_next_location_only_while_queue_has_candidates():
    """未検出時は残り地点がある間だけ次の地点へ進む."""
    context = TaskContext(target_objects=['cup'])
    context.grasp_goals.append(NavigationGoal('second', 1.0, 2.0, 0.0))
    state = SimpleNamespace(
        context=context,
        node=SimpleNamespace(get_logger=lambda: _Logger()),
    )

    assert RecogState._not_found_outcome(state) == 'next_location'
    context.grasp_goals.clear()
    assert RecogState._not_found_outcome(state) == 'failed'


def test_recog_uses_joint_pose_of_current_location():
    """Recogは固定姿勢ではなく設定済みの場所ごとの姿勢を使う."""
    operations = []
    logger = _Logger()
    context = TaskContext(
        current_grasp_goal=NavigationGoal(
            'pick_table',
            1.0,
            2.0,
            0.0,
            (('head_tilt_joint', math.radians(-40.0)),),
        )
    )
    state = SimpleNamespace(
        context=context,
        hsrif=SimpleNamespace(
            whole_body=SimpleNamespace(
                move_to_joint_positions=lambda positions, **kwargs: operations.append(
                    (positions, kwargs)
                )
            )
        ),
        node=SimpleNamespace(get_logger=lambda: logger),
    )

    assert RecogState._move_to_recognition_pose(state)
    assert operations == [
        ({'head_tilt_joint': math.radians(-40.0)}, {'sync': True})
    ]


def test_grasp_uses_apply_force_and_context_result():
    """把持姿勢をTaskContextから読みapply_forceで把持する."""
    operations = []
    context = TaskContext()
    context.set_grasp_result('cup', object(), 0.05)
    gripper = SimpleNamespace(
        command=lambda value: operations.append(('command', value)),
        apply_force=lambda **kwargs: operations.append(('force', kwargs)),
    )
    whole_body = SimpleNamespace(
        move_end_effector_pose=lambda *args, **kwargs: operations.append(
            ('pose', args, kwargs)
        ),
        move_end_effector_by_line=lambda *args, **kwargs: operations.append(
            ('line', args, kwargs)
        ),
        move_to_go=lambda **kwargs: operations.append(('go', kwargs)),
    )
    state = SimpleNamespace(
        context=context,
        hsrif=SimpleNamespace(gripper=gripper, whole_body=whole_body),
        node=SimpleNamespace(get_logger=lambda: _Logger()),
    )

    assert GraspState.execute(state, None) == 'succeeded'
    assert [operation[0] for operation in operations] == [
        'command',
        'pose',
        'line',
        'force',
        'go',
    ]
    assert operations[3][1] == {
        'effort': GRASP_FORCE,
        'delicate': False,
        'sync': True,
    }


def test_bring_obj_returns_success_and_logs_selected_object():
    """物体を人へ運んだ後、物体名を表示して正常終了する."""
    logger = _Logger()
    navigation_goals = []
    state = SimpleNamespace(
        context=TaskContext(recognized_object='apple'),
        nav=SimpleNamespace(
            nav_goal=lambda **kwargs: navigation_goals.append(kwargs) or True,
        ),
        node=SimpleNamespace(get_logger=lambda: logger),
    )

    assert BringObjState.execute(state, None) == 'succeeded'
    assert len(navigation_goals) == 1
    assert any('選んだ物体: apple' in message for message in logger.messages)


def test_prediction_result_converter_describes_order_as_text():
    """GPT結果と探索キューを読みやすい自然文へ変換する."""
    description = PredictionResultConverter.describe_order(
        {
            'target_object': 'りんご',
            'target_location': '丸テーブル',
            'confidence': 0.85,
            'pointing_direction': '東',
            'object_direction': '北東',
            'reason': '赤くて丸いため',
        },
        ['りんご', 'いちご'],
        ['丸テーブル', '棚'],
    )

    assert '「りんご」と推定しました' in description
    assert '確信度は85%です' in description
    assert '「丸テーブル」 → 「棚」の順に移動します' in description


def test_receive_order_logs_prediction_as_text():
    """ReceiveOrderがconverterで生成した自然文を表示する."""
    context = TaskContext()
    context.begin_order(
        ['りんご'],
        [NavigationGoal('丸テーブル', 1.0, 2.0, 0.5)],
        {
            'target_object': 'りんご',
            'target_location': '丸テーブル',
            'confidence': 0.9,
            'pointing_direction': '東',
            'object_direction': '東',
            'reason': '指差し方向と一致したため',
        },
    )
    logger = _Logger()
    state = SimpleNamespace(
        context=context,
        node=SimpleNamespace(get_logger=lambda: logger),
    )

    ReceiveOrderState._log_prediction_result(state)

    assert len(logger.messages) == 1
    assert 'GPT 認識・実行結果' in logger.messages[0]
    assert '「りんご」と推定しました' in logger.messages[0]
    assert '「丸テーブル」の順に移動します' in logger.messages[0]


def test_receive_order_logs_instruction_before_request():
    """人の指示を整形してGPT API呼び出し前に表示する."""
    events = []
    logger = SimpleNamespace(
        info=lambda message: events.append(('log', message)),
        error=lambda message: events.append(('error', message)),
    )
    state = SimpleNamespace(
        node=SimpleNamespace(get_logger=lambda: logger),
        hsrif=SimpleNamespace(
            whole_body=SimpleNamespace(
                move_to_joint_positions=lambda _: None,
            )
        ),
        instruction_provider=lambda: '  赤くて丸いもの  ',
        _capture_image=lambda: 'image',
        _request=lambda instruction, _image: events.append(
            ('request', instruction)
        ) or {},
        _validate_and_update=lambda _: None,
        _log_prediction_result=lambda: None,
        context=TaskContext(),
    )

    assert ReceiveOrderState.execute(state, None) == 'succeeded'
    instruction_index = next(
        index
        for index, event in enumerate(events)
        if event[0] == 'log' and '人からの指示' in event[1]
    )
    request_index = events.index(('request', '赤くて丸いもの'))
    assert '赤くて丸いもの' in events[instruction_index][1]
    assert instruction_index < request_index
