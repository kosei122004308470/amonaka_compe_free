"""借り物競争ステート間の分岐とデータ受け渡しを検証する."""

from types import SimpleNamespace

from amonaka_compe_free.states.grasp import GraspState
from amonaka_compe_free.states.recog import RecogState
from amonaka_compe_free.task_context import NavigationGoal
from amonaka_compe_free.task_context import TaskContext


class _Logger:
    """テスト中のログを破棄する."""

    def info(self, _):
        """INFOログを破棄する."""

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
        'effort': 2.0,
        'delicate': False,
        'sync': True,
    }
