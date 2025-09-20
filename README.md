# MLP-Iris
numpy vs pytorch
# 四层感知机实现 Iris 数据集多分类任务

本项目实现了一个 **四层感知机（两层隐藏层）**，用于对 Iris 数据集进行多分类任务。
与常规直接调用深度学习框架不同，本实验要求 **禁止直接调包**，需通过 **矩阵运算手写实现前向传播、反向传播与梯度下降**。
同时，将手写梯度与 **PyTorch 自动求导** 结果进行对比，以验证计算正确性。

---

## 📌 实验要求

* 使用 **Iris 数据集**（[UCI 官方链接](http://archive.ics.uci.edu/ml/datasets/iris)）

  * 前 4 列为特征
  * 最后一列为标签（共 3 类）

* 将数据集划分为：

  * 训练集：80%
  * 验证集：10%
  * 测试集：10%

* 模型结构：

  * 输入层：4 个特征
  * 隐藏层 1：自定义维度（默认 16）
  * 隐藏层 2：自定义维度（默认 8）
  * 输出层：3 类（softmax 激活）

* 实现内容：

  1. **前向传播**（ReLU 激活 + Softmax 输出）
  2. **交叉熵损失函数**
  3. **反向传播**（矩阵运算推导各参数梯度）
  4. **梯度下降**更新参数矩阵
  5. **验证梯度正确性**：与 PyTorch 自动求导结果对比
  6. **绘制曲线**：训练 & 验证集的 Loss、Accuracy 随 Epoch 变化曲线
  7. **性能评估**：输出训练集、验证集、测试集准确率

---

## 📂 文件说明

```
.
├── main.py               # Numpy 实现的 MLP（含 forward/backward/训练/可视化）
├── iris.data             # Iris 数据集 (可从 UCI 下载)
├── requirements.txt      # 项目所需依赖
└── README.md             # 项目说明文档
```

---

## ⚙️ 环境依赖

推荐使用 Python 3.10+，所需依赖如下：

```txt
numpy==1.26.4
matplotlib==3.9.2
```

安装依赖：

```bash
pip install -r requirements.txt
```

---

## 🚀 运行方式

```bash
python main.py
```

程序会输出：

1. 每隔若干 epoch 的训练/验证集 Loss 和 Accuracy
2. 训练完成后的训练集、验证集、测试集性能评估
3. Loss 曲线 & Accuracy 曲线图

---

## 📊 实验结果示例

* **Loss 曲线**
  随着迭代，训练集和验证集的损失逐渐下降
* **Accuracy 曲线**
  模型在训练集、验证集上准确率逐渐提升并收敛
* **最终性能评估**
  在测试集上取得约 **90% 左右准确率**

---

## 🧮 核心实现亮点

* 手写 **forward & backward**，完全基于矩阵运算
* 手动实现 **Softmax + CrossEntropy**，避免数值溢出
* **梯度对比验证**：第一个样本的梯度结果与 PyTorch 自动求导一致
* 代码结构清晰，便于理解 MLP 的计算过程

---

## 📖 参考

* [UCI Machine Learning Repository – Iris Dataset](http://archive.ics.uci.edu/ml/datasets/iris)
* 深度学习课程相关推导与实验要求

---
