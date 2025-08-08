import torch
import os
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import Counter

# シード値の設定
seed_value = 42
torch.manual_seed(seed_value)
np.random.seed(seed_value)
print(f"シード値 {seed_value} が設定されました。")

# デバイスの設定 (GPUが利用可能ならGPU、そうでなければCPU)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"計算に {device} を使用します。")

# データセットのルートディレクトリ
data_dir = './split_dataset'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

# 画像の変換処理
# データ拡張を強化
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# データローダーの作成
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# クラス数の取得
num_classes = len(train_dataset.classes)
class_names = train_dataset.classes
print(f"クラス数: {num_classes}")
print(f"クラス名: {class_names}")

# クラスの重み付け
class_counts = Counter(train_dataset.targets)
class_weights = torch.tensor([len(train_dataset) / class_counts[i] for i in range(num_classes)], dtype=torch.float32)
class_weights = class_weights.to(device)

# モデルの定義
model = models.densenet121(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

# 損失関数と最適化関数の定義
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学習の履歴を保存するためのリストを追加
train_losses = []
test_accuracies = []

# 学習
num_epochs = 15
print("\n学習を開始します...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_losses.append(running_loss / len(train_loader))
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {train_losses[-1]:.4f}')

    # テスト (学習中の進捗確認)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)
    print(f'Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {accuracy:.2f}%')

print('学習が完了しました。')

# テスト結果の集計とプロット
print("\n--- テスト結果の詳細 ---")
model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

total_samples = len(all_labels)
correct_predictions = (np.array(all_predictions) == np.array(all_labels)).sum()

overall_accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0

print(f"\n--- 全体の予測結果 ---")
print(f"総サンプル数: {total_samples}")
print(f"正解数: {correct_predictions}")
print(f"不正解数: {total_samples - correct_predictions}")
print(f"全体正答率: {overall_accuracy:.2f}%")

print(f"\n--- クラス別予測結果 ---")
print(f"{'クラス名':<20} | {'正解数':<10} | {'総数':<10} | {'正答率':<10}")
print(f"{''.join(['-']*20)}-+-{''.join(['-']*10)}-+-{''.join(['-']*10)}-+-{''.join(['-']*10)}")

class_accuracies = []
for i in range(num_classes):
    class_name = class_names[i]
    correct_count = ((np.array(all_predictions) == i) & (np.array(all_labels) == i)).sum()
    total_count = (np.array(all_labels) == i).sum()
    class_accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    class_accuracies.append(class_accuracy)
    print(f"{class_name:<20} | {correct_count:<10} | {total_count:<10} | {class_accuracy:.2f}%")

# プロット生成
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, marker='o', linestyle='-', color='blue')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.xticks(range(1, num_epochs + 1))

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), test_accuracies, marker='o', linestyle='-', color='green')
plt.title('Test Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.xticks(range(1, num_epochs + 1))

plt.tight_layout()
plt.savefig('densenet64_balanced_training_loss_accuracy.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(class_names, class_accuracies, color='skyblue')
plt.xlabel('Class Name')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy per Class')
plt.ylim(0, 100)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('densenet64_balanced_class_accuracy_bar.png')
plt.show()

cm = confusion_matrix(all_labels, all_predictions)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(num_classes + 2, num_classes + 2))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Proportion of Predictions'})
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Normalized by True Label)')
plt.savefig('densenet64_balanced_confusion_matrix_heatmap.png')
plt.show()
