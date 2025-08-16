# -*- coding: utf-8 -*-
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import numpy as np
from scipy.stats import pearsonr
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
        label = self.labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

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
    labels_list = []
    preds_list = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).float().view(-1, 1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            labels_list.extend(labels.cpu().numpy())
            preds_list.extend(outputs.cpu().numpy())

    val_loss /= len(loader.dataset)
    # 计算MSE然后手动开方得到RMSE
    val_mse = mean_squared_error(labels_list, preds_list)
    val_rmse = np.sqrt(val_mse)
    val_r2 = r2_score(labels_list, preds_list)
    val_pcc, _ = pearsonr(np.squeeze(labels_list), np.squeeze(preds_list))
    return val_loss, val_rmse, val_r2, val_pcc

# 自定义ResNet50模型
class ModifiedResNet50(models.ResNet):
    def __init__(self, *args, **kwargs):
        super(ModifiedResNet50, self).__init__(*args, **kwargs)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),  # Dropout层
            nn.Linear(self.fc.in_features, 1)  # 修改最后一层用于回归任务
        )

def main():
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    # 对目标值进行归一化处理（不需要）
    #scaler = MinMaxScaler()
    labels = pd.read_csv(labels_file)
    #labels.iloc[:, 1] = scaler.fit_transform(labels.iloc[:, 1].values.reshape(-1, 1))
    #labels.to_csv(labels_file, index=False)

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

    # 使用ResNet50模型
    model = ModifiedResNet50(models.resnet.Bottleneck, [3, 4, 6, 3])
    model.load_state_dict(models.resnet50(pretrained=True).state_dict(), strict=False)
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # 训练模型
    num_epochs = 200
    best_val_pcc = -float('inf')
    best_val_r2 = -float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_labels = []
        train_preds = []

        train_loader_tqdm = tqdm(train_loader, desc="Epoch {}/{}".format(epoch + 1, num_epochs), ncols=100)

        for images, labels in train_loader_tqdm:
            images = images.to(device)
            labels = labels.to(device).float().view(-1, 1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            train_labels.extend(labels.cpu().numpy())
            train_preds.extend(outputs.cpu().detach().numpy())

            train_loader_tqdm.set_postfix({"Loss": loss.item()})

        epoch_loss = running_loss / len(train_loader.dataset)
        # 计算MSE然后手动开方得到RMSE
        epoch_mse = mean_squared_error(train_labels, train_preds)
        epoch_rmse = np.sqrt(epoch_mse)
        epoch_r2 = r2_score(train_labels, train_preds)
        epoch_pcc, _ = pearsonr(np.squeeze(train_labels), np.squeeze(train_preds))
        print('Epoch {}/{}, Loss: {:.4f}, RMSE: {:.4f}, R²: {:.4f}, PCC: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss, epoch_rmse, epoch_r2, epoch_pcc))

        # 验证模型
        val_loss, val_rmse, val_r2, val_pcc = evaluate_model(model, val_loader, criterion, device)
        print('Validation Loss: {:.4f}, RMSE: {:.4f}, R²: {:.4f}, PCC: {:.4f}'.format(val_loss, val_rmse, val_r2, val_pcc))

        # 保存最好的模型
        if val_pcc > best_val_pcc:
            best_val_pcc = val_pcc
            torch.save(model.state_dict(), 'best_model909-ResNet.pth')

        scheduler.step()

    print('Training complete')

    # 测试模型
    model.load_state_dict(torch.load('best_model909-ResNet.pth'))
    model.eval()
    test_loss = 0.0
    test_labels = []
    test_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).float().view(-1, 1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            test_labels.extend(labels.cpu().numpy())
            test_preds.extend(outputs.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    # 计算MSE然后手动开方得到RMSE
    test_mse = mean_squared_error(test_labels, test_preds)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(test_labels, test_preds)
    test_pcc, _ = pearsonr(np.squeeze(test_labels), np.squeeze(test_preds))
    print('Test Loss: {:.4f}, RMSE: {:.4f}, R²: {:.4f}, PCC: {:.4f}'.format(test_loss, test_rmse, test_r2, test_pcc))
    residuals = np.array(test_preds) - np.array(test_labels)
    count_below_threshold = np.sum(np.abs(residuals) < 0.005)
    proportion_below_threshold = count_below_threshold / len(residuals)
    print('Proportion of residuals below 0.005: {:.2%}'.format(proportion_below_threshold))

    data = {
        'Test Loss': [test_loss],
        'RMSE': [test_rmse],
        'R²': [test_r2],
        'PCC': [test_pcc],
        'Proportion of residuals < 0.005': [proportion_below_threshold]
    }

    df = pd.DataFrame(data)

    # 保存DataFrame到CSV文件
    df.to_csv('test_metrics909-ResNet.csv', index=False)
    print("Metrics saved to 'test_metrics.csv'.")

if __name__ == '__main__':
    main()
