# 1. データセット構築について

`datasets_code` フォルダに保存されている以下の3つのコードを **順に実行** していきます。

1. 国別に収集された元データを地域別に整理  
   `split.py`
2. それを機械学習用に訓練データとテストデータに分割  
   `set_all.py`
3. 最終的に出来上がったデータセットの内容を確認  
   `check.py`

---

# 2. 学習・分析モデルの適用について

`learn_code` フォルダに保存されている以下のコードを **順に実行** します。

1. パラメーター調整  
   `learn_code/Patch_size.py` を実行し、適切なパラメータを実験に適用
2. 学習率と画像サイズの調整  
   `learn_code/hyperparameter_tuning.py` を `learn_code/run_experiments.sh` で実行
3. 最初の実験  
   `learn_code/densenet.py` を適用して実験
4. 重み付き学習の実験  
   最初の実験から変更を加え、`learn_code/densenet_balanced.py` を使用
