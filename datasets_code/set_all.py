import os
import shutil
import random

# シード値の設定
random.seed(42)

# 画像ファイルが格納されているベースディレクトリ
base_dir = '/Users/shiodatsubasa/データマイニング班/group/organized_dataset'

# 出力先のベースディレクトリ
output_split_dir = '/Users/shiodatsubasa/データマイニング班/group/split_dataset'
os.makedirs(output_split_dir, exist_ok=True)

# 分割比率
train_ratio = 0.8
test_ratio = 0.2

# 地域ごとの必要画像枚数（地域名: 枚数）
region_image_counts = {
    'Asia': 4206,
    'Middle East': 806,
    'Japan': 3840,
    'North America': 14536,
    'South America': 4125,
    'Europe': 18385,
    'Africa': 2103,
    'Oceania': 2262
}

# 'Asia（中東・日本除く）' フォルダが存在する場合は 'Asia' にリネーム
old_asia_path = os.path.join(base_dir, 'Asia（中東・日本除く）')
new_asia_path = os.path.join(base_dir, 'Asia')
if os.path.exists(old_asia_path) and os.path.isdir(old_asia_path) and not os.path.exists(new_asia_path):
    os.rename(old_asia_path, new_asia_path)
elif not os.path.exists(new_asia_path):
    print("Error: 'Asia' フォルダが見つかりません。")

for region, required_count in region_image_counts.items():
    region_path = os.path.join(base_dir, region)
    if os.path.exists(region_path) and os.path.isdir(region_path):
        all_images = [f for f in os.listdir(region_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
        random.shuffle(all_images)

        if len(all_images) < required_count:
            print(f"警告: {region} フォルダには {required_count} 枚必要ですが、{len(all_images)} 枚しか見つかりませんでした。利用可能な枚数で進行します。")
            selected_images = all_images
        else:
            selected_images = all_images[:required_count]

        # 学習用とテスト用のフォルダを作成
        train_dir = os.path.join(output_split_dir, 'train', region)
        test_dir = os.path.join(output_split_dir, 'test', region)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # 学習用とテスト用の画像リストを分割
        train_split = int(len(selected_images) * train_ratio)
        train_images = selected_images[:train_split]
        test_images = selected_images[train_split:]

        # 学習用画像をコピー
        for image in train_images:
            source_path = os.path.join(region_path, image)
            destination_path = os.path.join(train_dir, image)
            shutil.copy(source_path, destination_path)

        # テスト用画像をコピー
        for image in test_images:
            source_path = os.path.join(region_path, image)
            destination_path = os.path.join(test_dir, image)
            shutil.copy(source_path, destination_path)

        print(f"{region} フォルダを学習用とテスト用に分割しました ({len(train_images)}枚 (学習), {len(test_images)}枚 (テスト))")
    else:
        print(f"{region} フォルダが見つかりません。")

print("学習用とテスト用データセットの分割が完了しました。")