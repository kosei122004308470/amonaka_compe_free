# このリポジトリでやりたいこと
TOYOTA HSRに借り物競争的なタスクを行わせる。人から曖昧な言語指示と、正解オブジェクトの方向を指差すポーズがヒントとして与えられる。OpenAI APIでGPTに言語指示や指差しのポーズ画像、オブジェクトリストや部屋配置をプロンプトとして与え、お題の答えとなる物体を探しに行く。
# ステートマシン
```
Move2Human -> ReceiveOrder -> Move2Grasp -> Recog -> Grasp -> Move2Human
                                    |        |
                                    ----------
```
## Move2Human, Move2Grasp
ナビゲーションするステート
中身は事実上同一で、Move2Humanは行き先が固定（人の前）で、Move2Human.pyのなかで定数で定義する、Move2GraspはOpenAI APIの出力が格納されたキューからポップするる
## Recog
現行のrecogを基本的にそのまま使う
もし正解物体があれば把持に、もしなくて、行き先キューに１つ以上要素があればMove2Graspに戻る処理を追加する
なければFailed

## Grasp
現行のGraspをほぼそのまま使うが、gripper.apply_forceによる把持に変更する
## ReceiveOrder
今gpt_callなどで使われているプロンプトをそのまま使ってGPTに指差し方向推定、正解物体推定を行わせる
物体はリストとして、場所はキューとしてTaskContextに保管

# ステート間の情報共有
TaskContextというdataclassを別で定義して必ずそれぞれのステートに渡す形で共有する
正解物体リストや優先する把持場所リストなど