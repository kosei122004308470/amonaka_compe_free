"""
GPTの予測結果をリストまたは自然文へ変換する.

JSONファイルからリストを生成する場合の出力例
(
    ['strawberry', 'tomatocan', 'apple'],
    ['A', 'B', 'C']
)
"""

import json
from pathlib import Path
from typing import Iterable
from typing import Mapping


class PredictionResultConverter:
    """GPTの予測結果をステートで利用しやすい形式へ変換する."""

    def __init__(self, object_json_path: Path, point_json_path: Path):
        self.object_json_path = object_json_path
        self.point_json_path = point_json_path

    def load(self, json_path: Path) -> dict:
        with json_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def get_o_no1_objects(self) -> list[str]:
        result = self.load(self.object_json_path)
        return [item["object"] for item in result["o_no1"]]

    def get_p_no1_points(self) -> list[str]:
        result = self.load(self.point_json_path)
        return [item["point"] for item in result["p_no1"]]

    @staticmethod
    def describe_order(
        result: Mapping[str, object],
        target_objects: Iterable[str],
        search_locations: Iterable[str],
    ) -> str:
        """GPTの認識結果とロボットの探索計画を自然文へ変換する."""
        target_object = str(result.get('target_object', '不明'))
        target_location = str(result.get('target_location', '不明'))
        pointing_direction = str(result.get('pointing_direction', '不明'))
        object_direction = str(result.get('object_direction', '不明'))
        reason = str(result.get('reason', '理由なし'))
        try:
            confidence = f'{float(result.get("confidence", 0.0)):.0%}'
        except (TypeError, ValueError):
            confidence = '不明'

        objects = '、'.join(f'「{name}」' for name in target_objects)
        locations = ' → '.join(
            f'「{name}」' for name in search_locations
        )
        if not objects:
            objects = 'なし'
        if not locations:
            locations = 'なし'

        return (
            f'GPTは指差し方向を「{pointing_direction}」、対象物の方向を'
            f'「{object_direction}」と認識し、最も可能性の高い物体を'
            f'「{target_object}」と推定しました。確信度は{confidence}です。'
            f'最初の探索場所は「{target_location}」です。'
            f'判断理由は「{reason}」です。\n'
            f'ロボットは対象候補{objects}を探すため、'
            f'{locations}の順に移動します。'
        )


def main() -> tuple[list[str], list[str]]:
    base_dir = Path(__file__).resolve().parent
    object_json_path = base_dir / "result_object_prediction.json"
    point_json_path = base_dir / "result_point_prediction.json"

    converter = PredictionResultConverter(object_json_path, point_json_path)

    return converter.get_o_no1_objects(), converter.get_p_no1_points()


if __name__ == "__main__":
    main()
