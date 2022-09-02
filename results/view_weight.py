#%%
import torch
from compressai.models.elic import ELIC

net = ELIC()
checkpoint = torch.load('pretrained/elic/6/skip_best.pth.tar')
net.load_state_dict(checkpoint['state_dict'])
#%%
parm={}
for name,parameters in net.named_parameters():
    if "space" in name:
        print(name,':',parameters.size())
        parm[name]=parameters.detach().numpy()
# %%
from matplotlib import pyplot as plt
w = parm['contextiter._sub_modules_space.0.cross_conv.weight']
# %%

