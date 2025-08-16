
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
labels_file = '/home/chuannong1/ZLY/2024/IMF/大理石纹等级/labels906.csv'

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
    val_rmse = mean_squared_error(labels_list, preds_list, squared=False)
    val_r2 = r2_score(labels_list, preds_list)
    val_pcc, _ = pearsonr(np.squeeze(labels_list), np.squeeze(preds_list))
    return val_loss, val_rmse, val_r2, val_pcc

# 自定义VGG模型
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(num_channels // reduction_ratio, num_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        return self.sigmoid(out).view(x.size(0), x.size(1), 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(num_channels, reduction_ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class ModifiedVGG(nn.Module):
    def __init__(self):
        super(ModifiedVGG, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        self.cbams = nn.ModuleList([CBAM(64), CBAM(128), CBAM(256), CBAM(512), CBAM(512)])

        # 冻结VGG特征提取部分的权重
        for param in self.vgg.features.parameters():
            param.requires_grad = False
        
        # 修改分类器部分
        num_features = self.vgg.classifier[-1].in_features
        self.vgg.classifier = nn.Sequential(
            *list(self.vgg.classifier.children())[:-1],
            nn.Dropout(0.5),
            nn.Linear(num_features, 1)  # 假设是一个二分类问题
        )

    def forward(self, x):
        # 应用特征提取器和CBAM模块，注意特征提取区段的切分需要匹配VGG结构
        sections = [5, 10, 17, 24, 31]  # 这些是VGG每个卷积块的结束层索引
        start = 0
        for section, cbam in zip(sections, self.cbams):
            x = self.vgg.features[start:section](x)
            x = cbam(x)
            start = section

        x = self.vgg.features[start:](x)
        x = self.vgg.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.vgg.classifier(x)
        return x


def main():
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    # 使用DenseNet模型
    model = ModifiedVGG()
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

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", ncols=100)

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
        epoch_rmse = mean_squared_error(train_labels, train_preds, squared=False)
        epoch_r2 = r2_score(train_labels, train_preds)
        epoch_pcc, _ = pearsonr(np.squeeze(train_labels), np.squeeze(train_preds))
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, RMSE: {epoch_rmse:.4f}, R²: {epoch_r2:.4f}, PCC: {epoch_pcc:.4f}')

        # 验证模型
        val_loss, val_rmse, val_r2, val_pcc = evaluate_model(model, val_loader, criterion, device)
        print(f'Validation Loss: {val_loss:.4f}, RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}, PCC: {val_pcc:.4f}')

        # 保存最好的模型
        if val_pcc > best_val_pcc:
            best_val_pcc = val_pcc
            torch.save(model.state_dict(), r'/home/chuannong1/ZLY/2024/IMF/大理石纹等级/VGG/best_model906-VGGCBAM.pth')

        scheduler.step()

    print('Training complete')

    # 测试模型
    model.load_state_dict(torch.load(r'/home/chuannong1/ZLY/2024/IMF/大理石纹等级/VGG/best_model906-VGGCBAM.pth'))
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
    test_rmse = mean_squared_error(test_labels, test_preds, squared=False)
    test_r2 = r2_score(test_labels, test_preds)
    test_pcc, _ = pearsonr(np.squeeze(test_labels), np.squeeze(test_preds))
    print(f'Test Loss: {test_loss:.4f}, RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}, PCC: {test_pcc:.4f}')
    residuals = np.array(test_preds) - np.array(test_labels)
    count_below_threshold = np.sum(np.abs(residuals) < 0.005)
    proportion_below_threshold = count_below_threshold / len(residuals)
    print(f'Proportion of residuals below 0.005: {proportion_below_threshold:.2%}')

    data = {
        'Test Loss': [test_loss],
        'RMSE': [test_rmse],
        'R²': [test_r2],
        'PCC': [test_pcc],
        'Proportion of residuals < 0.005': [proportion_below_threshold]
    }

    df = pd.DataFrame(data)

    # 保存DataFrame到CSV文件
    df.to_csv('/home/chuannong1/ZLY/2024/IMF/大理石纹等级/VGG/test_metrics-VGGCBAM.csv', index=False)
    print("Metrics saved to 'test_metrics.csv'.")



if __name__ == '__main__':
    main()
