# -*- coding: utf-8 -*-
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
        img_path_jpg = os.path.join(self.img_dir, "{}.jpg".format(img_name))
        img_path_png = os.path.join(self.img_dir, "{}.png".format(img_name))

        if os.path.exists(img_path_jpg):
            img_path = img_path_jpg
        elif os.path.exists(img_path_png):
            img_path = img_path_png
        else:
            print("File not found: {} or {}".format(img_path_jpg, img_path_png))
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
labels_file = '/home/chuannong1/ZLY/2024/IMF/大理石纹等级/labelswsl909.csv'

# 评估模型函数
def evaluate_model(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            
            # 获取预测类别和概率
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    val_loss /= len(loader.dataset)
    
    # 计算分类评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(all_labels, all_preds)
    
    # 计算ROC-AUC（多分类情况下需要特殊处理）
    try:
        # 尝试计算多分类的ROC-AUC
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except ValueError:
        # 如果出现错误（例如只有一个类别），则设为0
        roc_auc = 0
    
    return val_loss, accuracy, precision, recall, f1, kappa, roc_auc

# Transformer模块实现
class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=8, num_layers=3):
        super(Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, 
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.transformer_encoder(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# 自定义ResNet50+Transformer模型
class ModifiedResNet50(nn.Module):
    def __init__(self, num_classes=3, transformer_hidden_dim=512, transformer_output_dim=256):
        super(ModifiedResNet50, self).__init__()
        # ResNet50 backbone
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = self.resnet.fc.in_features  # 2048
        self.resnet.fc = nn.Identity()  # 移除原始的全连接层
        
        # Transformer模块
        self.transformer = Transformer(
            input_dim=num_ftrs, 
            hidden_dim=transformer_hidden_dim, 
            output_dim=transformer_output_dim
        )
        
        # 最终分类层
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs + transformer_output_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # ResNet特征提取
        resnet_features = self.resnet(x)  # (batch_size, 2048)
        resnet_features = resnet_features.view(resnet_features.size(0), -1)
        
        # Transformer处理
        # 为Transformer添加序列维度
        transformer_input = resnet_features.unsqueeze(1)  # (batch_size, 1, 2048)
        transformer_output = self.transformer(transformer_input)  # (batch_size, 1, transformer_output_dim)
        transformer_output = transformer_output.squeeze(1)  # (batch_size, transformer_output_dim)
        
        # 特征融合
        combined_features = torch.cat((resnet_features, transformer_output), dim=1)
        
        # 分类输出
        output = self.classifier(combined_features)
        return output

def main():
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    # 加载数据集
    dataset = PorkFatDataset(img_dir=img_dir, labels_file=labels_file, transform=train_transform)
    
    # 获取类别数量
    labels = pd.read_csv(labels_file)
    num_classes = len(labels.iloc[:, 1].unique())
    print("Number of classes: {}".format(num_classes))

    # 划分训练集、验证集和测试集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # 使用ResNet50+Transformer模型
    model = ModifiedResNet50(
        num_classes=num_classes, 
        transformer_hidden_dim=512, 
        transformer_output_dim=256
    )
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # 训练模型
    num_epochs = 200
    best_val_f1 = -float('inf')
    
    # 早停机制参数
    patience = 20  # 耐心值：连续20个epoch没有改善就停止
    early_stop_counter = 0  # 早停计数器
    min_delta = 0.001  # 最小改善阈值

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_labels = []
        train_preds = []
        train_probs = []

        train_loader_tqdm = tqdm(train_loader, desc="Epoch {}/{}".format(epoch + 1, num_epochs), ncols=100)

        for images, labels in train_loader_tqdm:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            
            # 获取预测类别和概率
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            train_labels.extend(labels.cpu().numpy())
            train_preds.extend(preds.cpu().detach().numpy())
            train_probs.extend(probs.cpu().detach().numpy())

            train_loader_tqdm.set_postfix({"Loss": loss.item()})

        epoch_loss = running_loss / len(train_loader.dataset)
        
        # 计算训练集评估指标
        train_accuracy = accuracy_score(train_labels, train_preds)
        train_precision = precision_score(train_labels, train_preds, average='weighted', zero_division=0)
        train_recall = recall_score(train_labels, train_preds, average='weighted', zero_division=0)
        train_f1 = f1_score(train_labels, train_preds, average='weighted', zero_division=0)
        train_kappa = cohen_kappa_score(train_labels, train_preds)
        
        try:
            train_roc_auc = roc_auc_score(train_labels, train_probs, multi_class='ovr')
        except ValueError:
            train_roc_auc = 0
        
        print('Epoch {}/{}, Loss: {:.4f}, Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, Kappa: {:.4f}, ROC-AUC: {:.4f}'.format(
            epoch + 1, num_epochs, epoch_loss, train_accuracy, train_precision, train_recall, train_f1, train_kappa, train_roc_auc))

        # 验证模型
        val_loss, val_accuracy, val_precision, val_recall, val_f1, val_kappa, val_roc_auc = evaluate_model(model, val_loader, criterion, device)
        print('Validation Loss: {:.4f}, Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, Kappa: {:.4f}, ROC-AUC: {:.4f}'.format(
            val_loss, val_accuracy, val_precision, val_recall, val_f1, val_kappa, val_roc_auc))

        # 早停机制和模型保存逻辑
        if val_f1 > best_val_f1 + min_delta:
            best_val_f1 = val_f1
            early_stop_counter = 0  # 重置计数器
            torch.save(model.state_dict(), 'best_model_ResNet_TRS_classification.pth')
            print('New best model saved with F1 score: {:.4f}'.format(best_val_f1))
        else:
            early_stop_counter += 1
            print('Early stopping counter: {}/{}'.format(early_stop_counter, patience))
            
            # 如果连续patience个epoch没有改善，则停止训练
            if early_stop_counter >= patience:
                print('Early stopping triggered! No improvement for {} epochs.'.format(patience))
                print('Best validation F1 score: {:.4f}'.format(best_val_f1))
                break

        scheduler.step()

    print('Training complete')

    # 测试模型
    model.load_state_dict(torch.load('best_model_ResNet_TRS_classification.pth'))
    model.eval()
    test_loss = 0.0
    test_labels = []
    test_preds = []
    test_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            
            # 获取预测类别和概率
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            test_labels.extend(labels.cpu().numpy())
            test_preds.extend(preds.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    
    # 计算测试集评估指标
    test_accuracy = accuracy_score(test_labels, test_preds)
    test_precision = precision_score(test_labels, test_preds, average='weighted', zero_division=0)
    test_recall = recall_score(test_labels, test_preds, average='weighted', zero_division=0)
    test_f1 = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
    test_kappa = cohen_kappa_score(test_labels, test_preds)
    
    try:
        test_roc_auc = roc_auc_score(test_labels, test_probs, multi_class='ovr')
    except ValueError:
        test_roc_auc = 0
    
    # 计算混淆矩阵
    cm = confusion_matrix(test_labels, test_preds)
    print('Test Loss: {:.4f}, Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, Kappa: {:.4f}, ROC-AUC: {:.4f}'.format(
        test_loss, test_accuracy, test_precision, test_recall, test_f1, test_kappa, test_roc_auc))
    print('Confusion Matrix:')
    print(cm)

    data = {
        'Test Loss': [test_loss],
        'Accuracy': [test_accuracy],
        'Precision': [test_precision],
        'Recall': [test_recall],
        'F1 Score': [test_f1],
        'Cohen\'s Kappa': [test_kappa],
        'ROC-AUC': [test_roc_auc]
    }

    df = pd.DataFrame(data)

    # 保存DataFrame到CSV文件
    df.to_csv('test_metrics_ResNet_TRS_classification.csv', index=False)
    print("Metrics saved to 'test_metrics_ResNet_TRS_classification.csv'.")

if __name__ == '__main__':
    main()