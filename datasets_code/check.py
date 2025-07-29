import os

data_dir = './split_dataset' 

# 確認したい地域フォルダ名
regions = [
    'Africa',
    'Asia',
    'Europe',
    'Japan',
    'Middle East',
    'North America',
    'Oceania',
    'South America'
]

print(f"データセットディレクトリ: {data_dir}\n")

total_images_count = 0

for split in ['train', 'test']:
    split_path = os.path.join(data_dir, split)
    if not os.path.exists(split_path) or not os.path.isdir(split_path):
        print(f"'{split}' フォルダが見つかりません: {split_path}")
        continue

    print(f"--- {split} データセット ---")
    split_total_count = 0
    for region in regions:
        region_path = os.path.join(split_path, region)
        if os.path.exists(region_path) and os.path.isdir(region_path):
            image_count = 0
            for filename in os.listdir(region_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    image_count += 1
            print(f"  {region} フォルダ内の画像枚数: {image_count}枚")
            split_total_count += image_count
        else:
            print(f"  {region} フォルダが見つかりません: {region_path}")
    print(f"  {split} データセットの合計画像枚数: {split_total_count}枚\n")
    total_images_count += split_total_count

print(f"全データセットの合計画像枚数: {total_images_count}枚")
