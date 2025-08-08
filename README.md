1.データセット構築について
datasets_codeのフォルダに保存されている以下の３つのコードを順に実行していく。
「1.国別に収集された元データを地域別に整理し (split.py)」
「2.それを機械学習用に訓練データとテストデータに分割し (set_all.py)」
「3.最終的に出来上がったデータセットの内容を確認する (check.py)」という3つのステップで構成する。

2.学習・分析モデルの適用について
learn_codeのフォルダに保存されている。以下のつのコードを順に実行していく。
「1.パラメーター調整についてはlearn_code/Patch_size.pyを実行することで、適切なパラーメータを実験に適用させていった」
「2.学習率と画像サイズについては、learn_code/hyperparameter_tuning.pyをlearn_code/run_experiments.shで動かすことで、調整を行なった。」
「3.最初の実験では、learn_code/densenet.pyを適用して、実験を行った」
「4.最初の実験から変更を行い、’重み’を付けた学習を行うために、learn_code/densenet_balanced.pyを使用した。」