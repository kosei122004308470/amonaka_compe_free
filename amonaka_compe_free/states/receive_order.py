#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""人の指示とカメラ画像から探索対象・把持場所を決定するステート."""

import ast
import base64
import json
import math
from pathlib import Path
from typing import Callable

import yaml
from openai import OpenAI
from yasmin import State

from ..pointing_pic import CameraCapture
from ..prediction_result_converter import PredictionResultConverter
from ..task_context import NavigationGoal
from ..task_context import TaskContext


MODEL = 'gpt-5.6-luna'
REASONING_EFFORT = 'medium'
IMAGE_TOPIC = '/head_rgbd_sensor/rgb/image_color'
DEFAULT_INSTRUCTION = '泡がでるもの'
# HSR の arm_lift_joint だけは直動関節なので YAML の値をメートルのまま
# 用いる。その他の joint は YAML では度数で記載されている。
LINEAR_JOINTS = {'arm_lift_joint'}


def default_instruction_provider() -> str:
    """標準入力を使わない動作確認用の固定指示を返す."""
    return DEFAULT_INSTRUCTION


class ReceiveOrderState(State):
    """GPT に人の指示を解釈させ、TaskContext を更新する."""

    def __init__(
        self,
        node,
        hsrif,
        context: TaskContext,
        client=None,
        camera=None,
        instruction_provider: Callable[[], str] | str | None = None,
        config_dir: Path | None = None,
    ):
        """
        ステートを初期化する.

        ``client``、``camera``、``instruction_provider`` はテスト時に
        差し替えられる。実機では省略すると OpenAI と ROS カメラを使う。
        """
        super().__init__(outcomes=['succeeded', 'failed'])
        self.node = node
        self.hsrif = hsrif
        self.context = context
        self.client = (
            client
            if client is not None
            else OpenAI(default_headers={'Accept-Encoding': 'gzip'})
        )
        self.camera = camera if camera is not None else CameraCapture(
            image_topic=IMAGE_TOPIC
        )
        self._owns_camera = camera is None
        if instruction_provider is None:
            self.instruction_provider = default_instruction_provider
        elif callable(instruction_provider):
            self.instruction_provider = instruction_provider
        elif isinstance(instruction_provider, str):
            self.instruction_provider = lambda: instruction_provider
        else:
            raise TypeError(
                'instruction_provider は関数または文字列で指定してください。'
            )
        self.config_dir = (
            Path(config_dir)
            if config_dir is not None
            else self._default_config_dir()
        )
        self._pdf_file_id = None

    @staticmethod
    def _default_config_dir() -> Path:
        """ソースツリーまたは ROS インストール先の設定ディレクトリを返す."""
        source_dir = Path(__file__).resolve().parents[2] / 'config'
        if source_dir.is_dir():
            return source_dir
        try:
            from ament_index_python.packages import (
                get_package_share_directory,
            )
            return (
                Path(get_package_share_directory('amonaka_compe_free'))
                / 'config'
            )
        except (ImportError, LookupError):
            return source_dir

    @staticmethod
    def _system_prompt() -> str:
        """gpt_call.py の system_prompt を原文のまま読み込む."""
        prompt_path = Path(__file__).resolve().parents[1] / 'gpt_call.py'
        tree = ast.parse(prompt_path.read_text(encoding='utf-8'))
        for statement in tree.body:
            if not isinstance(statement, ast.Assign):
                continue
            if any(
                isinstance(target, ast.Name)
                and target.id == 'system_prompt'
                for target in statement.targets
            ):
                prompt = ast.literal_eval(statement.value)
                if isinstance(prompt, str):
                    return prompt
        raise ValueError('gpt_call.py に system_prompt がありません。')

    def _load_inputs(self) -> tuple[dict, dict, dict]:
        """既存 GPT スクリプトと同じ3つの YAML を読み込む."""
        paths = (
            self.config_dir / 'object_list.yaml',
            self.config_dir / 'carrobo_move.yaml',
            self.config_dir / 'carrobo_human.yaml',
        )
        try:
            return tuple(
                yaml.safe_load(path.read_text(encoding='utf-8')) or {}
                for path in paths
            )
        except (OSError, yaml.YAMLError) as error:
            raise ValueError(
                f'設定ファイルを読み込めません: {error}'
            ) from error

    def _pdf_id(self):
        """物体リスト PDF をアップロードし、その ID を返す."""
        if self._pdf_file_id is not None:
            return self._pdf_file_id
        pdf_path = self.config_dir / 'object_list.pdf'
        try:
            with pdf_path.open('rb') as pdf_file:
                uploaded = self.client.files.create(
                    file=pdf_file,
                    purpose='user_data',
                )
        except (OSError, AttributeError) as error:
            raise ValueError(
                f'物体リスト PDF をアップロードできません: {error}'
            ) from error
        self._pdf_file_id = uploaded.id
        return self._pdf_file_id

    def _capture_image(self) -> str:
        """最新カメラ画像を JPEG Base64 へ変換する."""
        jpeg_data = self.camera.capture()
        if jpeg_data is None:
            raise ValueError(
                'カメラ画像の取得に失敗しました。'
            )
        return base64.b64encode(jpeg_data).decode('utf-8')

    def _request(self, instruction: str, image_base64: str) -> dict:
        """既存 gpt_call.py と同じ入力形式で Responses API を呼ぶ."""
        object_list, map_info, carrobo_human = self._load_inputs()
        user_input = f"""
【人間の指示】

{instruction}


【オブジェクトリスト】

{object_list}


【マップ情報】

{map_info}


【CarroboとHumanの位置情報】

{carrobo_human}
"""
        response = self.client.responses.create(
            model=MODEL,
            reasoning={'effort': REASONING_EFFORT},
            instructions=self._system_prompt(),
            input=[
                {
                    'role': 'user',
                    'content': [
                        {'type': 'input_text', 'text': user_input},
                        {
                            'type': 'input_image',
                            'image_url': (
                                f'data:image/jpeg;base64,{image_base64}'
                            ),
                        },
                        {'type': 'input_file', 'file_id': self._pdf_id()},
                    ],
                }
            ],
        )
        try:
            output_text = response.output_text
            result = json.loads(output_text)
        except (AttributeError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(
                'GPT の出力が指定された JSON 形式ではありません: '
                f'{error}'
            ) from error
        if not isinstance(result, dict):
            raise ValueError(
                'GPT の出力 JSON がオブジェクトではありません。'
            )
        return result

    @staticmethod
    def _leaf_names(value: object) -> set[str]:
        """カテゴリ辞書から物体名（葉のキー）だけを抽出する."""
        if not isinstance(value, dict):
            return set()
        names = set()
        for key, child in value.items():
            if isinstance(child, dict):
                names.update(ReceiveOrderState._leaf_names(child))
            else:
                names.add(str(key))
        return names

    def _validate_and_update(self, result: dict) -> None:
        """GPT 結果を検証し、検証成功時だけ TaskContext を更新する."""
        required = {
            'target_object',
            'target_location',
            'confidence',
            'pointing_direction',
            'object_direction',
            'reason',
            'search_priority',
        }
        missing = required.difference(result)
        if missing:
            raise ValueError(
                f'GPT 出力に必須キーがありません: {sorted(missing)}'
            )

        object_list, map_info, _ = self._load_inputs()
        object_names = self._leaf_names(object_list)
        target_object = result['target_object']
        if target_object not in object_names:
            raise ValueError(
                f'未知の target_object です: {target_object}'
            )

        try:
            confidence = float(result['confidence'])
        except (TypeError, ValueError) as error:
            raise ValueError(
                'confidence が数値ではありません。'
            ) from error
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                'confidence は 0.0 から 1.0 の範囲で指定してください。'
            )

        priority = result['search_priority']
        if not isinstance(priority, list):
            raise ValueError('search_priority が配列ではありません。')

        entries = [
            {
                'object': target_object,
                'location': result['target_location'],
            }
        ]
        for item in priority:
            if not isinstance(item, dict):
                raise ValueError(
                    'search_priority の要素がオブジェクトではありません。'
                )
            for key in ('object', 'location'):
                if key not in item or not isinstance(item[key], str):
                    raise ValueError(
                        f'search_priority に {key} がありません。'
                    )
            if item['object'] not in object_names:
                raise ValueError(
                    f'未知の候補物体です: {item["object"]}'
                )
            entries.append(item)

        goals = []
        seen_locations = set()
        for entry in entries:
            location = entry['location']
            if location in seen_locations:
                continue
            seen_locations.add(location)
            location_config = map_info.get(location)
            if not isinstance(location_config, dict):
                raise ValueError(f'未知の探索場所です: {location}')
            try:
                map_pose = location_config['map']
                joint_config = location_config['joint']
                if not isinstance(map_pose, dict) or not isinstance(
                    joint_config, dict
                ):
                    raise TypeError
                x = float(map_pose['x'])
                y = float(map_pose['y'])
                yaw = math.radians(float(map_pose['theta']))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f'探索場所 {location} の map 設定が不正です。'
                ) from error
            if not all(math.isfinite(value) for value in (x, y, yaw)):
                raise ValueError(
                    f'探索場所 {location} の map 設定が不正です。'
                )

            joint_positions = []
            try:
                for joint_name, raw_value in joint_config.items():
                    if not isinstance(joint_name, str):
                        raise TypeError
                    value = float(raw_value)
                    if not math.isfinite(value):
                        raise ValueError
                    if joint_name not in LINEAR_JOINTS:
                        value = math.radians(value)
                    joint_positions.append((joint_name, value))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f'探索場所 {location} の joint 設定が不正です。'
                ) from error
            if not joint_positions:
                raise ValueError(
                    f'探索場所 {location} の joint 設定がありません。'
                )
            goals.append(
                NavigationGoal(location, x, y, yaw, tuple(joint_positions))
            )

        target_objects = []
        for entry in entries:
            if entry['object'] not in target_objects:
                target_objects.append(entry['object'])
        self.context.begin_order(target_objects, goals, result)

    def _log_prediction_result(self) -> None:
        """GPTの認識結果と採用した探索計画を自然文で表示する."""
        description = PredictionResultConverter.describe_order(
            self.context.order_result or {},
            self.context.target_objects,
            (goal.name for goal in self.context.grasp_goals),
        )
        self.node.get_logger().info(
            '\n========== GPT 認識・実行結果 ==========\n'
            f'{description}\n'
            '===================================='
        )

    def execute(self, _) -> str:
        """人の指示を受け、GPT 結果を TaskContext に保存する."""
        self.node.get_logger().info('Executing state ReceiveOrder')
        try:
            self.hsrif.whole_body.move_to_joint_positions(
                {
                    'head_pan_joint': 0.0,
                    'head_tilt_joint': 0.3,
                }
            )
            instruction = self.instruction_provider()
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError('人間の指示が空です。')
            instruction = instruction.strip()
            self.node.get_logger().info(
                '\n========== 人からの指示 ==========\n'
                f'{instruction}\n'
                '===================================='
            )
            result = self._request(instruction, self._capture_image())
            self._validate_and_update(result)
            self._log_prediction_result()
        except Exception as error:
            self.node.get_logger().error(
                f'指示の受信に失敗しました: {error}'
            )
            return 'failed'
        self.node.get_logger().info(
            f'探索対象を {self.context.target_objects}、'
            f'探索場所を {len(self.context.grasp_goals)} 件設定しました。'
        )
        return 'succeeded'

    def close(self) -> None:
        """このステートが所有するカメラノードを解放する."""
        if self._owns_camera and self.camera is not None:
            self.camera.destroy_node()
