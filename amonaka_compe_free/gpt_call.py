from openai import OpenAI
import yaml

client = OpenAI()

with open("/home/carrobo2026/hma2_ws/src/amonaka_compe_free/config/object_list.yaml", "r", encoding="utf-8") as f:
    object_list = yaml.safe_load(f)

with open("/home/carrobo2026/hma2_ws/src/amonaka_compe_free/config/carrobo_move.yaml", "r", encoding="utf-8") as f:
    map_info = yaml.safe_load(f)

pdf_path = "/home/carrobo2026/hma2_ws/src/amonaka_compe_free/config/object_list.pdf"

with open(pdf_path, "rb") as f:
    pdf_file = client.files.create(
        file=f,
        purpose="user_data"
    )
print("PDF upload OK")


system_prompt = """
あなたはロボットの物体探索を支援するAIです。

人間からの曖昧な指示、オブジェクトリスト、
指差し画像、object_list.pdf、マップ情報を統合して、
ロボットが探すべき対象物と探索場所を決定してください。


【最重要ルール】

対象物を100%確定できない場合でも、
「特定不能」で探索を終了してはいけません。

候補となる物体を比較し、
最も可能性が高い物体を必ず1つ選択してください。

その物体を「第1候補」として、
ロボットはその物体を優先して探索します。

ただし、確信度は正確に設定してください。


【候補の優先順位】

対象物を決定するときは、以下の情報を総合的に評価してください。

優先順位は基本的に、

1. 指差し方向・指差し位置
2. 指差し画像内の物体の位置
3. 指差し画像とPDFの物体写真の外観一致
4. 人間の指示内容
5. オブジェクトの色
6. オブジェクトの形状・大きさ・特徴
7. マップ上の位置関係

です。

特に指差し方向が得られる場合は、
単純な色一致よりも指差し方向を優先してください。


【複数候補がある場合】

例えば、

「赤いものを持ってきて」

という指示に対して、

- りんご
- いちご
- 赤い皿
- 赤いボウル

が候補になった場合、

「赤い物体が複数あるため特定不能」

として終了してはいけません。

それぞれの候補について、

- 指示との一致
- 指差し方向との一致
- 指差し画像との一致
- PDF写真との一致
- 形状
- 色
- 大きさ
- マップ上の位置

を比較してください。

そして最も可能性の高い候補を第1候補として選択してください。


【探索戦略】

対象物を完全に特定できない場合でも、
最も可能性の高い候補から探索を開始できるようにしてください。

例えば、

第1候補：りんご 0.55
第2候補：いちご 0.30
第3候補：赤い皿 0.15

の場合、

target_objectには「りんご」を設定してください。

ロボットはまず「りんご」が存在する可能性のある場所へ向かいます。


【探索失敗時】

第1候補を探索して対象物が見つからなかった場合は、
第2候補へ移行できるように、
候補を順位付きで出力してください。

したがって、候補を1つだけ出力するのではなく、
可能性の高い順番にすべての候補を出力してください。


【重要】

- object_listに存在しない物体を候補にしない。
- マップに存在しない場所を作らない。
- 根拠のない候補を追加しない。
- 情報が不足している場合でも、最も可能性の高い候補を1つ選ぶ。
- 「特定不能」をtarget_objectにしない。
- confidenceが低くても、最も可能性の高い候補をtarget_objectにする。


【出力形式】

以下のJSON形式で出力してください。

{
  "target_object": "最も可能性の高い物体",
  "target_location": "その物体を探索する場所",
  "confidence": 0.0,
  "reason": "第1候補を選択した理由",

  "search_priority": [
    {
      "rank": 1,
      "object": "第1候補",
      "location": "探索場所",
      "confidence": 0.0,
      "reason": "候補として最も可能性が高い理由"
    },
    {
      "rank": 2,
      "object": "第2候補",
      "location": "探索場所",
      "confidence": 0.0,
      "reason": "第2候補とした理由"
    },
    {
      "rank": 3,
      "object": "第3候補",
      "location": "探索場所",
      "confidence": 0.0,
      "reason": "第3候補とした理由"
    }
  ]
}


【confidence】

0.90～1.00：
ほぼ確実に対象物を特定できる。

0.70～0.89：
かなり可能性が高い。

0.50～0.69：
候補として有力だが不確実。

0.30～0.49：
不確実だが、探索候補として利用できる。

0.00～0.29：
かなり不確実。

confidenceが低い場合でも、
target_objectを「特定不能」にしてはいけません。


【最終目的】

このAIの目的は、
「対象物を完全に言い当てること」ではなく、

「最も可能性の高い対象物を決定し、
その場所へロボットを移動させ、
見つからなければ次の候補を探索する」

ことです。
"""


# 人間が発した指示
instruction = input("人間の指示を入力してください：")

user_input = f"""
人間の指示：
{instruction}

オブジェクトリスト：
{object_list}

マップ：
{map_info}
"""

response = client.responses.create(
    model="gpt-5.6-luna",
    reasoning={
        "effort": "medium"
    },
    instructions=system_prompt,
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": user_input
                },
                {
                    "type": "input_file",
                    "file_id": pdf_file.id
                }
            ]
        }
    ]
)

print("GPT:")
print(response.output_text)
