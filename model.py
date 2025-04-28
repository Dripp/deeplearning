# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from srm_filter_kernel import all_normalized_hpf_list
from MPNCOV import CovpoolLayer, SqrtmLayer, TriuvecLayer
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import os

class TLU(nn.Module):
    def __init__(self, threshold):
        super(TLU, self).__init__()
        self.threshold = threshold

    def forward(self, input):
        output = torch.clamp(input, min=-self.threshold, max=self.threshold)
        return output

class HPF(nn.Module):
    def __init__(self):
        super(HPF, self).__init__()
        all_hpf_list_5x5 = []
        for hpf_item in all_normalized_hpf_list:
            if hpf_item.shape[0] == 3:
                hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')
            all_hpf_list_5x5.append(hpf_item)

        hpf_weight = np.array(all_hpf_list_5x5)
        hpf_weight = nn.Parameter(torch.from_numpy(hpf_weight).view(30, 1, 5, 5), requires_grad=False)

        self.hpf = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=False)
        self.hpf.weight = hpf_weight
        self.tlu = TLU(1.0)

        # 调试：打印滤波器权重
        print("SRM Filter Weights (first filter):")
        print(self.hpf.weight[0].detach().cpu().numpy())

    def forward(self, input, debug=False, epoch=1, batch_idx=0, output_dir="srm_features"):
        output = self.hpf(input)
        output = self.tlu(output)

        # 调试：打印 SRM 输出统计信息并可视化特征图
        if debug and batch_idx == 0:
            print(f"SRM Output (epoch {epoch}, batch {batch_idx}):")
            print(f"Mean: {output.mean().item():.4f}, Std: {output.std().item():.4f}")
            print(f"Min: {output.min().item():.4f}, Max: {output.max().item():.4f}")

            # 可视化 SRM 特征图（仅保存第一个样本的前 5 个特征图）
            os.makedirs(output_dir, exist_ok=True)
            feature_maps = output[0].detach().cpu().numpy()  # 第一个样本，形状 (30, H, W)
            for i in range(min(5, feature_maps.shape[0])):  # 前 5 个特征图
                plt.figure(figsize=(5, 5))
                plt.imshow(feature_maps[i], cmap='gray')
                plt.title(f"SRM Feature Map {i}")
                plt.axis('off')
                plt.savefig(os.path.join(output_dir, f"epoch_{epoch}_feature_map_{i}.png"))
                plt.close()

        return output

class Net(nn.Module):
    def __init__(self, use_srm=True):
        super(Net, self).__init__()
        self.use_srm = use_srm
        if self.use_srm:
            self.hpf = HPF()
        self.resnet = resnet18(weights=None)
        in_channels = 30 if self.use_srm else 1  # 如果不使用 SRM，则输入通道为 1
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)

    def forward(self, input, debug=False, epoch=1, batch_idx=0):
        if self.use_srm:
            output = self.hpf(input, debug=debug, epoch=epoch, batch_idx=batch_idx)
        else:
            output = input  # 直接使用原始输入
        output = self.resnet(output)
        return output

def initWeights(module):
    if type(module) == nn.Conv2d:
        if module.weight.requires_grad:
            nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')
        if module.bias is not None and module.bias.requires_grad:
            nn.init.constant_(module.bias.data, val=0.0)
    elif type(module) == nn.BatchNorm2d:
        if module.weight.requires_grad:
            nn.init.normal_(module.weight.data, mean=1.0, std=0.02)
        if module.bias.requires_grad:
            nn.init.constant_(module.bias.data, val=0.0)
    elif type(module) == nn.Linear:
        nn.init.normal_(module.weight.data, mean=0.0, std=0.01)
        if module.bias.requires_grad:
            nn.init.constant_(module.bias.data, val=0.0)