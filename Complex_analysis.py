import torch
from types import SimpleNamespace
import yaml
from models.Mafe_embedding import Mafe_embedding

with open('/home/wangdongzhihan/sjn/ODGNet-main/cfgs/PCN_models/isomap_embedding.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
model_config = SimpleNamespace(**cfg['model'])
model = Mafe_embedding(model_config).cuda()

total_params = sum(p.numel() for p in model.parameters())
print(f'Total parameters: {total_params:,}')  # 带千分位更易读from thop import profile

from thop import profile

# 假设 model 为你的点云补全网络，输入为 [B, 3, N]
import torch
input = torch.randn(1, 2048, 3).cuda()
flops, params = profile(model,  inputs=(input,), verbose=False)

print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
print(f"Parameters: {params / 1e6:.2f} M")