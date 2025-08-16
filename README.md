# 猪肉脂肪分析深度学习项目

本项目使用深度学习方法对猪肉脂肪进行分类和回归分析，实现了多种深度学习模型和注意力机制。

## 功能特点

- 多种深度学习架构：ResNet、VGG、DenseNet、EfficientNet
- 多种注意力机制：CBAM、SE、TRS
- 支持分类和回归任务
- 完整的评估指标体系
- 早停机制

## 环境要求

- Python 3.8+
- PyTorch 1.8+
- torchvision
- scikit-learn
- pandas
- numpy
- PIL
- tqdm

## 安装方法

```bash
git clone https://github.com/DeepBLUP/lmage-texture-recognition.git
cd pork-fat-analysis
pip install -r requirements.txt
```

## 使用方法

### 训练模型

```python
# 分类任务示例
python train/train_classification.py --model resnet --attention cbam

# 回归任务示例
python train/train_regression.py --model vgg --attention se
```

### 使用预训练模型

```python
# 示例代码
from models.resnet.resnet_base import ModifiedResNet50
import torch

# 加载模型
model = ModifiedResNet50(num_classes=5)
model.load_state_dict(torch.load("path/to/model.pth"))
model.eval()
```

## 项目结构

- `models/`: 模型定义
  - `resnet/`: ResNet相关模型
  - `vgg/`: VGG相关模型
  - `densenet/`: DenseNet相关模型
  - `efficientnet/`: EfficientNet相关模型
- `utils/`: 工具函数
  - `dataset.py`: 数据集处理
  - `metrics.py`: 评估指标
- `train/`: 训练脚本
- `examples/`: 使用示例
- `data/`: 数据目录（需自行添加数据）

## 许可证

本项目采用MIT许可证 - 详见LICENSE文件