import torch
import os
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt  # プロットのために追加
from sklearn.metrics import confusion_matrix
import seaborn as sns

# シード値の設定
seed_value = 42
torch.manual_seed(seed_value)
np.random.seed(seed_value)
print(f"シード値 {seed_value} が設定されました。")

# デバイスの設定 (CPUを使用)
device = torch.device('cpu')
print(f"計算に {device} を使用します。")

# データセットのルートディレクトリ
data_dir = '/Users/isamimuua/info3dm/G4プログラム/split_dataset'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

# 画像の変換処理
transform = transforms.Compose([
    transforms.Resize((64, 64)),  #-------------------------------------------------
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# データローダーの作成
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# クラス数の取得
num_classes = len(train_dataset.classes)
class_names = train_dataset.classes
print(f"クラス数: {num_classes}")
print(f"クラス名: {class_names}")

# モデル定義
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * 16 * 16, num_classes)

    def forward(self, x):
        x = self.maxpool1(self.relu1(self.conv1(x)))
        x = self.maxpool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = self.fc(x)
        return x

model = SimpleCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 履歴を保存するリスト
train_losses = []
test_accuracies = []

# エポック数を x に設定
num_epochs = 30 #--------------------------------------------------------
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

# テスト結果の集計と表示
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
correct_predictions = 0
class_correct = {i: 0 for i in range(num_classes)}
class_total = {i: 0 for i in range(num_classes)}
class_accuracies = []

for i in range(total_samples):
    true_label = all_labels[i]
    pred_label = all_predictions[i]
    class_total[true_label] += 1
    if true_label == pred_label:
        correct_predictions += 1
        class_correct[true_label] += 1

overall_accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
print(f"\n--- 全体の予測結果 ---")
print(f"総サンプル数: {total_samples}")
print(f"正解数: {correct_predictions}")
print(f"不正解数: {total_samples - correct_predictions}")
print(f"全体正答率: {overall_accuracy:.2f}%")

print(f"\n--- クラス別予測結果 ---")
print(f"{'クラス名':<20} | {'正解数':<10} | {'総数':<10} | {'正答率':<10}")
print(f"{'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
for i in range(num_classes):
    class_name = class_names[i]
    correct_count = class_correct[i]
    total_count = class_total[i]
    class_accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    class_accuracies.append(class_accuracy)
    print(f"{class_name:<20} | {correct_count:<10} | {total_count:<10} | {class_accuracy:.2f}%")

# プロット
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
plt.show()

# クラス別正答率の棒グラフ
plt.figure(figsize=(10, 6))
plt.bar(class_names, class_accuracies, color='skyblue')
plt.xlabel('Class Name')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy per Class')
plt.ylim(0, 100)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# --- 3. 混同行列のヒートマップのプロット ---
print("\n--- 混同行列のヒートマップ ---")

cm = confusion_matrix(all_labels, all_predictions)

# 各行を合計で割ることで正規化
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(num_classes + 2, num_classes + 2))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Proportion of Predictions'})
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Normalized by True Label)')
plt.show()