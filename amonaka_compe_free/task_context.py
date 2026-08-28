"""借り物競争タスクのステート間で共有するデータ."""

from collections import deque
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Iterable


@dataclass(frozen=True)
class NavigationGoal:
    """検証済みのナビゲーション候補を表す."""

    name: str
    x: float
    y: float
    yaw: float
    # carrobo_move.yaml の joint を ROS 単位へ変換して保持する。
    # tuple にしておくことで、ステート間で誤って書き換えない。
    joint_positions: tuple[tuple[str, float], ...] = ()

    def recognition_joint_positions(self) -> dict[str, float]:
        """認識時に whole_body へ渡す関節角度を返す."""
        return dict(self.joint_positions)


@dataclass
class TaskContext:
    """GPT の推定結果と現在の探索位置をステート間で共有する."""

    target_objects: list[str] = field(default_factory=list)
    grasp_goals: deque[NavigationGoal] = field(default_factory=deque)
    current_grasp_goal: NavigationGoal | None = None
    order_result: dict | None = None
    recognized_object: str | None = None
    grasp_pose: Any | None = None
    grasp_approach: float | None = None

    def begin_order(
        self,
        target_objects: Iterable[str],
        grasp_goals: Iterable[NavigationGoal],
        order_result: dict,
    ) -> None:
        """新しい指示と探索候補を保存し、前回の認識結果を消去する."""
        self.target_objects = list(target_objects)
        self.grasp_goals = deque(grasp_goals)
        self.current_grasp_goal = None
        self.order_result = dict(order_result)
        self.clear_grasp_result()

    def pop_grasp_goal(self) -> NavigationGoal | None:
        """次の把持場所候補をFIFOキューから取り出す."""
        if not self.grasp_goals:
            self.current_grasp_goal = None
            return None
        self.current_grasp_goal = self.grasp_goals.popleft()
        self.clear_grasp_result()
        return self.current_grasp_goal

    @property
    def has_pending_grasp_goals(self) -> bool:
        """未探索の把持場所候補が残っているか返す."""
        return bool(self.grasp_goals)

    def set_grasp_result(
        self,
        object_name: str,
        grasp_pose: Any,
        grasp_approach: float,
    ) -> None:
        """認識した物体と把持動作用の値を保存する."""
        self.recognized_object = object_name
        self.grasp_pose = grasp_pose
        self.grasp_approach = float(grasp_approach)

    def clear_grasp_result(self) -> None:
        """前回の認識・把持結果を消去する."""
        self.recognized_object = None
        self.grasp_pose = None
        self.grasp_approach = None
