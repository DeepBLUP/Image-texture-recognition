import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F

# 设置随机种子
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 自定义Dataset类
class PorkFatDataset(Dataset):
    def __init__(self, img_dir, labels_file, transform=None, scaler=None):
        self.img_dir = img_dir
        self.labels = pd.read_csv(labels_file)
        self.transform = transform
        self.scaler = scaler
        
        # Convert continuous labels to classification labels (0, 1, 2, 3, 4)
        # Assuming labels are in range [1, 5], convert to [0, 4]
        self.labels.iloc[:, 1] = (self.labels.iloc[:, 1] - 1).astype(int)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = self.labels.iloc[idx, 0]
        img_path_jpg = os.path.join(self.img_dir, f"{img_name}.jpg")
        img_path_png = os.path.join(self.img_dir, f"{img_name}.png")

        if os.path.exists(img_path_jpg):
            img_path = img_path_jpg
        elif os.path.exists(img_path_png):
            img_path = img_path_png
        else:
            print(f"File not found: {img_path_jpg} or {img_path_png}")
            return None

        image = Image.open(img_path).convert("RGB")
        label = self.labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

# 数据预处理
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 数据集路径
img_dir = '/home/chuannong1/ZLY/2024/IMF/datasets/test1'
labels_file = '/home/chuannong1/ZLY/2024/IMF/大理石纹等级/labels906.csv'

# 评估模型函数
def evaluate_model(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    labels_list = []
    preds_list = []
    probs_list = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            
            # Get probabilities and predictions
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            labels_list.extend(labels.cpu().numpy())
            preds_list.extend(preds.cpu().numpy())
            probs_list.extend(probs.cpu().numpy())

    val_loss /= len(loader.dataset)
    
    # Calculate classification metrics
    accuracy = accuracy_score(labels_list, preds_list)
    precision = precision_score(labels_list, preds_list, average='weighted', zero_division=0)
    recall = recall_score(labels_list, preds_list, average='weighted', zero_division=0)
    f1 = f1_score(labels_list, preds_list, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(labels_list, preds_list)
    
    # Calculate ROC-AUC for multiclass
    try:
        roc_auc = roc_auc_score(labels_list, probs_list, multi_class='ovr', average='weighted')
    except:
        roc_auc = 0.0
    
    return val_loss, accuracy, precision, recall, f1, kappa, roc_auc, labels_list, preds_list, probs_list

# 自定义EfficientNet模型
class ModifiedEfficientNet(nn.Module):
    def __init__(self, num_classes=5):
        super(ModifiedEfficientNet, self).__init__()
        # 加载预训练的EfficientNet模型，使用weights参数替代deprecated的pretrained
        self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        # 冻结卷积层的参数，防止在训练过程中更新（可选）
        for param in self.efficientnet.features.parameters():
            param.requires_grad = False

        # 获取原始EfficientNet分类器的输入特征数
        num_features = self.efficientnet.classifier[-1].in_features

        # 修改分类器以适用于分类任务
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.efficientnet(x)

def main():
    # 设置设备
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据集
    dataset = PorkFatDataset(img_dir=img_dir, labels_file=labels_file, transform=train_transform)

    # 划分训练集、验证集和测试集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # 使用修改后的EfficientNet模型
    num_classes = 5  # 假设有5个分类等级
    model = ModifiedEfficientNet(num_classes=num_classes)
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # 训练模型
    num_epochs = 200
    best_val_f1 = -float('inf')
    patience = 20
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_labels = []
        train_preds = []

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", ncols=100)

        for images, labels in train_loader_tqdm:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_labels.extend(labels.cpu().numpy())
            train_preds.extend(preds.cpu().numpy())

            train_loader_tqdm.set_postfix({"Loss": loss.item()})

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = accuracy_score(train_labels, train_preds)
        epoch_f1 = f1_score(train_labels, train_preds, average='weighted', zero_division=0)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, F1: {epoch_f1:.4f}')

        # 验证模型
        val_loss, val_accuracy, val_precision, val_recall, val_f1, val_kappa, val_roc_auc, _, _, _ = evaluate_model(model, val_loader, criterion, device)
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, Kappa: {val_kappa:.4f}, ROC-AUC: {val_roc_auc:.4f}')

        # 早停和保存最好的模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), r'/home/chuannong1/ZLY/2024/IMF/大理石纹等级/eff/best_model906-effNet-classification.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        scheduler.step()

    print('Training complete')

    # 测试模型
    model.load_state_dict(torch.load(r'/home/chuannong1/ZLY/2024/IMF/大理石纹等级/eff/best_model906-effNet-classification.pth'))
    test_loss, test_accuracy, test_precision, test_recall, test_f1, test_kappa, test_roc_auc, test_labels, test_preds, test_probs = evaluate_model(model, test_loader, criterion, device)
    
    print(f'Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}, Kappa: {test_kappa:.4f}, ROC-AUC: {test_roc_auc:.4f}')
    
    # 计算混淆矩阵
    cm = confusion_matrix(test_labels, test_preds)
    print(f'Confusion Matrix:\n{cm}')

    # 保存结果
    data = {
        'Test Loss': [test_loss],
        'Accuracy': [test_accuracy],
        'Precision': [test_precision],
        'Recall': [test_recall],
        'F1': [test_f1],
        'Kappa': [test_kappa],
        'ROC-AUC': [test_roc_auc]
    }

    df = pd.DataFrame(data)
    df.to_csv('/home/chuannong1/ZLY/2024/IMF/大理石纹等级/eff/test_metrics-effNet-classification.csv', index=False)
    
    # 保存混淆矩阵
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv('/home/chuannong1/ZLY/2024/IMF/大理石纹等级/eff/confusion_matrix-effNet-classification.csv', index=False)
    
    print("Classification metrics and confusion matrix saved.")

if __name__ == '__main__':
    main()