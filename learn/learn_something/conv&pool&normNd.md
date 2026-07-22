## 约定输入张量格式：
[Batch, Channel, 维度1, 维度2, 维度3]
B：batch size；C：通道数；后面全部是空间 / 时序维度
1D：只存在 1 条轴线（时序、一维序列）
2D：平面（高度 H × 宽度 W，图像、二维网格流场）
3D：立方体（深度 D × H × W；视频时序 + 空间、三维场）
---
## Conv1d / Conv2d / Conv3d 区别
### conv1d
```python
nn.Conv1d(in_channels, out_channels, kernel_size)
# 输入 shape: [B, C, L]
# L：序列长度（唯一一维）
```
卷积核：沿单个长度维度滑动
kernel_size = 整数，代表一维窗口长度
✅ 使用场景：
一维时序信号、传感器序列、文本 token 序列、一维波形。
例：风速一维时间序列、音频波形。
### conv2d
```python
nn.Conv2d(in_channels, out_channels, kernel_size)
# 输入 shape: [B, C, H, W]
# H高度，W宽度（二维平面）
```
卷积核在 H、W 两个方向滑动
kernel_size 可以写 3 正方形，或 (kh, kw) 矩形
✅ 使用场景：
普通图像、二维空间网格、卫星二维快照。
### conv3d
```python
nn.Conv3d(in_channels, out_channels, kernel_size)
# 输入 shape: [B, C, D, H, W]
# D：深度/时序维度
```
卷积核同时在 D, H, W 三个方向滑动
✅ 使用场景：
视频（D = 时间帧）、三维体数据、时空立方体。
---
## BatchNorm1d / BatchNorm2d / BatchNorm3d
BatchNorm 本质：对每个通道单独归一化
区别只在于：哪些维度被当做 “样本维度” 用来统计均值 / 方差
### BatchNorm1d
```python
nn.BatchNorm1d(num_features=C)
# 支持两种合法shape：
# ① [B, C] 简单向量
# ② [B, C, L] 一维序列
```
统计方式：
在 B、L 维度上求均值方差，保持通道 C 独立。
✅搭配：Conv1d、全连接层输出
### BatchNorm2d
```python
nn.BatchNorm2d(num_features=C)
# 输入 shape: [B, C, H, W]
```
统计方式：
在 B、H、W 全部像素上聚合，每个通道独立归一化。
✅搭配：Conv2d（图像 / 二维网格标准搭配）
### BatchNorm3d
```python
nn.BatchNorm3d(num_features=C)
# 输入 shape: [B, C, D, H, W]
```
统计方式：
B、D、H、W 一起聚合求均值方差。
✅搭配：Conv3d

### 核心口诀 BN：
ConvNd 后面紧跟 BatchNormNd！
Conv1d → BN1d
Conv2d → BN2d
Conv3d → BN3d
---
## AvgPool1d / AvgPool2d / AvgPool3d & AdaptiveAvgPool
普通 AvgPool：指定池化窗口大小、步长，输出尺寸自动计算；
AdaptiveAvgPool（自适应池化）：直接指定想要的输出尺寸，框架自动算出 kernel/stride，非常好用。

### AvgPool1d / AdaptiveAvgPool1d
输入：[B,C,L]
在长度 L 维度做平均池化
```python
# 自适应，指定输出长度=8
nn.AdaptiveAvgPool1d(8)
```
用途：一维序列压缩长度
### AvgPool2d / AdaptiveAvgPool2d
输入：[B,C,H,W]
在 H、W 二维空间池化
```python
# 输出固定 [4,4] 特征图
nn.AdaptiveAvgPool2d((4,4))
# 最常用：全局平均池化 GAP，压缩到1×1
nn.AdaptiveAvgPool2d((1,1))
```
### AvgPool3d / AdaptiveAvgPool3d
输入：[B,C,D,H,W]
D、H、W 三维同时池化
### 池化选择准则
池化 Nd 和 ConvNd 维度保持一致：
Conv1d 输出 → Pool1d
Conv2d 输出 → Pool2d
Conv3d 输出 → Pool3d
---