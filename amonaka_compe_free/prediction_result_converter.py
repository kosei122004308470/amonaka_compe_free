"""
GPTモデルが出力した2つのJSONファイルからリストを生成する.

出力例
(
    ['strawberry', 'tomatocan', 'apple'],
    ['A', 'B', 'C']
)
"""

import json
from pathlib import Path


class PredictionResultConverter:
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


def main() -> tuple[list[str], list[str]]:
    base_dir = Path(__file__).resolve().parent
    object_json_path = base_dir / "result_object_prediction.json"
    point_json_path = base_dir / "result_point_prediction.json"

    converter = PredictionResultConverter(object_json_path, point_json_path)

    return converter.get_o_no1_objects(), converter.get_p_no1_points()


if __name__ == "__main__":
    main()
