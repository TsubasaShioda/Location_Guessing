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
from torch.optim.lr_scheduler import StepLR
import argparse
import json

def train_model(args):
    # シード値の設定
    seed_value = 42
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    print(f"シード値 {seed_value} が設定されました。")

    # デバイスの設定
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"計算に {device} を使用します。")

    # 結果保存ディレクトリの作成
    output_dir = os.path.join(args.output_dir, f"lr_{args.learning_rate}_size_{args.image_size}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"結果は {output_dir} に保存されます。")

    # データセットのルートディレクトリ
    data_dir = '/mnt/split_dataset'
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    # 画像の変換処理
    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # データローダーの作成
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # クラスの重み付け
    class_counts = Counter(train_dataset.targets)
    class_weights = torch.tensor([len(train_dataset) / class_counts[i] for i in range(len(train_dataset.classes))], dtype=torch.float32)
    class_weights = class_weights.to(device)

    # モデルの定義
    model = models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, len(train_dataset.classes))
    model = model.to(device)

    # 損失関数と最適化関数
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    # 学習
    print("\n学習を開始します...")
    train_losses, test_accuracies = [], []
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        scheduler.step()
        train_losses.append(running_loss / len(train_loader))
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        print(f'Epoch [{epoch+1}/{args.epochs}], Loss: {train_losses[-1]:.4f}, Accuracy: {accuracy:.2f}%')

    print('学習が完了しました。')

    # 結果の評価と保存
    evaluate_and_save_results(model, test_loader, train_dataset.classes, device, output_dir, args.epochs, train_losses, test_accuracies)

def evaluate_and_save_results(model, test_loader, class_names, device, output_dir, num_epochs, train_losses, test_accuracies):
    print("\n--- テスト結果の詳細 ---")
    model.eval()
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 全体精度
    overall_accuracy = (np.array(all_predictions) == np.array(all_labels)).mean() * 100
    print(f"\n全体正答率: {overall_accuracy:.2f}%")

    # クラス別精度
    class_accuracies = {}
    for i, name in enumerate(class_names):
        correct_count = ((np.array(all_predictions) == i) & (np.array(all_labels) == i)).sum()
        total_count = (np.array(all_labels) == i).sum()
        accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
        class_accuracies[name] = accuracy
        print(f"{name:<20} | 正答率: {accuracy:.2f}%")

    # 結果をJSONファイルに保存
    results = {
        'overall_accuracy': overall_accuracy,
        'class_accuracies': class_accuracies,
        'train_losses': train_losses,
        'test_accuracies': test_accuracies
    }
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)

    # プロット生成
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), test_accuracies, marker='o', color='green')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()

    # 混同行列
    cm = confusion_matrix(all_labels, all_predictions)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(len(class_names) + 2, len(class_names) + 2))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Densenet Hyperparameter Tuning')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='学習率')
    parser.add_argument('--image-size', type=int, default=64, help='画像サイズ')
    parser.add_argument('--epochs', type=int, default=30, help='エポック数')
    parser.add_argument('--batch-size', type=int, default=16, help='バッチサイズ')
    parser.add_argument('--output-dir', type=str, default='./tuning_results', help='結果の保存先ディレクトリ')
    
    args = parser.parse_args()
    train_model(args)
