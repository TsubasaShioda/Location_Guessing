import torch
import os
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# LoRAを適用するために必要なライブラリ (peft) をインポートします
# 事前にインストールが必要です: pip install peft transformers
try:
    from peft import get_peft_model, LoraConfig, TaskType
except ImportError:
    print("PEFTライブラリがインストールされていません。")
    print("ターミナルで `pip install peft transformers` を実行してください。")
    exit()

# データセットのルートディレクトリ
# こちらはご自身の環境に合わせて修正してください
data_dir = '/Users/e235727/info3dm/group_work/split_dataset'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

# 画像の変換処理（224x224にリサイズし、正規化）
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# データローダーの作成
batch_size = 32
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# クラス数の取得
num_classes = len(train_dataset.classes)
print(f"クラス数: {num_classes}")
print(f"クラス名: {train_dataset.classes}")

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"使用デバイス: {device}")

# ------------------- LoRAの組み込み（修正箇所） -------------------

# 1. ベースとなるDenseNet121モデルを読み込み、出力層を差し替え
from torchvision.models import densenet121, DenseNet121_Weights
weights = DenseNet121_Weights.DEFAULT
model = densenet121(weights=weights)
model.classifier = nn.Linear(model.classifier.in_features, num_classes)

# 2. LoRAを適用する対象の層を動的に見つける
#    モデル内の全てのConv2d層の名前を取得します
all_conv2d_layers = []
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        all_conv2d_layers.append(name)

#   効率的なファインチューニングのため、モデルの後半の層を対象にします。
#   ここでは最後の10個のConv2d層をLoRAのターゲットとします。
if len(all_conv2d_layers) > 10:
    target_modules = all_conv2d_layers[-10:]
else:
    target_modules = all_conv2d_layers

print("\nLoRAを適用する層 (target_modules):")
print(target_modules)

# 3. LoRAの設定を作成
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=target_modules,  # 動的に見つけた層の名前を使用
    lora_dropout=0.1,
    bias="none",
    #task_type=TaskType.IMAGE_CLASSIFICATION,
)

# 4. モデルにLoRAを適用
lora_model = get_peft_model(model, lora_config)
lora_model = lora_model.to(device)

# 5. 学習対象のパラメータ数を確認
print("\nLoRA適用後のモデル:")
lora_model.print_trainable_parameters()
# ----------------------------------------------------

# 損失関数と最適化関数の定義
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lora_model.parameters(), lr=0.001)

# 学習とリアルタイム可視化の準備
train_losses = []
test_accuracies = []

plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('LoRA Fine-tuning on DenseNet121')

num_epochs = 10
for epoch in range(num_epochs):
    # --- 学習フェーズ ---
    lora_model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = lora_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    # --- 評価フェーズ ---
    lora_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = lora_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # --- リアルタイム描画 ---
    ax1.clear()
    ax2.clear()
    ax1.plot(train_losses, label='Training Loss', marker='o')
    ax2.plot(test_accuracies, label='Test Accuracy', color='green', marker='o')
    ax1.set_title('Training Loss')
    ax2.set_title('Test Accuracy')
    ax1.set_xlabel('Epoch')
    ax2.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax2.set_ylabel('Accuracy (%)')
    ax1.grid(True)
    ax2.grid(True)
    ax1.legend()
    ax2.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.pause(0.1)

plt.ioff()
print("\n学習が完了しました。")
plt.show()

# --- 学習済みLoRAアダプターの保存 ---
output_dir = "lora_densenet_adapter"
lora_model.save_pretrained(output_dir)