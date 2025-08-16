
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

        if self.scaler:
            self.labels.iloc[:, 1] = self.scaler.transform(self.labels.iloc[:, 1].values.reshape(-1, 1))
        
        # 确保标签从0开始
        self.unique_labels = sorted(self.labels.iloc[:, 1].unique())
        self.label_map = {label: idx for idx, label in enumerate(self.unique_labels)}
        print(f"原始标签值: {self.unique_labels}")
        print(f"映射后标签值: {list(self.label_map.values())}")

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
        original_label = self.labels.iloc[idx, 1]
        # 将原始标签映射到从0开始的索引
        label = self.label_map[original_label]

        if self.transform:
            image = self.transform(image)

        # 对于分类任务，确保标签是整数类型
        return image, torch.tensor(int(label), dtype=torch.long)

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
def evaluate_model(model, dataloader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # 获取预测类别
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
            # 获取概率用于ROC-AUC计算
            probabilities = torch.softmax(outputs, dim=1)
            all_probabilities.extend(probabilities.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    
    # 计算分类指标
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(all_labels, all_predictions)
    
    # 计算ROC-AUC（多分类）
    try:
        if num_classes == 2:
            roc_auc = roc_auc_score(all_labels, [prob[1] for prob in all_probabilities])
        else:
            roc_auc = roc_auc_score(all_labels, all_probabilities, multi_class='ovr', average='weighted')
    except ValueError:
        roc_auc = 0.0
    
    return avg_loss, accuracy, precision, recall, f1, kappa, roc_auc, all_labels, all_predictions

# 自定义VGG模型
class ModifiedVGG(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedVGG, self).__init__()
        # 加载预训练的VGG16模型，使用weights参数替代deprecated的pretrained
        self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        
        # 冻结卷积层参数
        for param in self.vgg.features.parameters():
            param.requires_grad = False
        
        # 修改分类器，适应分类任务
        self.vgg.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)  # 输出类别数量
        )
    
    def forward(self, x):
        return self.vgg(x)

def main():
    # 设置设备
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据集
    dataset = PorkFatDataset(img_dir=img_dir, labels_file=labels_file, transform=train_transform)
    
    # 获取类别数量
    num_classes = len(dataset.unique_labels)
    print(f"类别数量: {num_classes}")

    # 划分训练集、验证集和测试集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # 为验证集和测试集设置不同的变换
    val_dataset.dataset.transform = val_test_transform
    test_dataset.dataset.transform = val_test_transform

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # 使用ModifiedVGG模型
    model = ModifiedVGG(num_classes=num_classes)
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 分类任务使用交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 训练模型
    num_epochs = 50
    best_val_f1 = 0
    patience = 10
    patience_counter = 0

    print("开始训练...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_labels = []
        all_predictions = []

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", ncols=100)

        for images, labels in train_loader_tqdm:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            train_loader_tqdm.set_postfix({"Loss": loss.item()})

        # 计算训练指标
        train_loss = running_loss / len(train_loader)
        train_accuracy = accuracy_score(all_labels, all_predictions)
        train_f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}")

        # 验证模型
        val_loss, val_accuracy, val_precision, val_recall, val_f1, val_kappa, val_roc_auc, _, _ = evaluate_model(
            model, val_loader, criterion, device, num_classes)
        print(f"  Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")

        # 早停机制（基于F1分数）
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), r'/home/chuannong1/ZLY/2024/IMF/大理石纹等级/VGG/best_model906-VGG.pth')
            patience_counter = 0
            print(f"  新的最佳模型已保存，F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停触发，在第 {epoch+1} 轮停止训练")
                break

        scheduler.step()
        print()

    print('Training complete')

    # 测试模型
    print("开始测试...")
    model.load_state_dict(torch.load(r'/home/chuannong1/ZLY/2024/IMF/大理石纹等级/VGG/best_model906-VGG.pth'))
    test_loss, test_accuracy, test_precision, test_recall, test_f1, test_kappa, test_roc_auc, test_labels, test_predictions = evaluate_model(
        model, test_loader, criterion, device, num_classes)

    print(f"测试结果:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print(f"  Test Precision: {test_precision:.4f}")
    print(f"  Test Recall: {test_recall:.4f}")
    print(f"  Test F1: {test_f1:.4f}")
    print(f"  Test Kappa: {test_kappa:.4f}")
    print(f"  Test ROC-AUC: {test_roc_auc:.4f}")
    
    # 计算混淆矩阵
    cm = confusion_matrix(test_labels, test_predictions)
    print(f"\n混淆矩阵:\n{cm}")

    # 保存测试指标到CSV文件
    test_metrics = {
        'Test Loss': [test_loss],
        'Test Accuracy': [test_accuracy],
        'Test Precision': [test_precision],
        'Test Recall': [test_recall],
        'Test F1': [test_f1],
        'Test Kappa': [test_kappa],
        'Test ROC-AUC': [test_roc_auc]
    }

    test_df = pd.DataFrame(test_metrics)
    test_df.to_csv('/home/chuannong1/ZLY/2024/IMF/大理石纹等级/VGG/test_metrics-VGG.csv', index=False)
    print("测试指标已保存到 test_metrics-VGG.csv")
    
    # 保存混淆矩阵
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv('/home/chuannong1/ZLY/2024/IMF/大理石纹等级/VGG/confusion_matrix-VGG.csv', index=False)
    print("混淆矩阵已保存到 confusion_matrix-VGG.csv")



if __name__ == '__main__':
    main()
