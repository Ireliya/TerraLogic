import subprocess

# res = subprocess.call(['./scripts/coop/main.sh caltech101 rn50_ep50 end 16 1 False', '-c', 'ls -al'], shell=True)
from model.hivit import HiViT

from thop import profile
from thop import clever_format
import torch

x = torch.rand(1,3,224,224)

model = HiViT(
    embed_dim=768, depths=[4, 4, 36], num_heads=12, stem_mlp_ratio=3., in_chans=3, mlp_ratio=4.,
    num_classes=25,
    ape=True, rpe=False,)

flops, params = profile(model, inputs=(x, ))
# print(flops, params)
macs, params = clever_format([flops, params], "%.3f")
print(macs,params)


