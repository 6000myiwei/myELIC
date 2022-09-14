#%%
import torch
from compressai.models.elic_ada5 import ELIC_ADA

net = ELIC_ADA()
checkpoint = torch.load('pretrained/elic_ada/6/mu_pooling_2.pth.tar')
net.load_state_dict(checkpoint['state_dict'])
#%%
parm={}
for name,parameters in net.named_parameters():
    if "space" in name:
        print(name,':',parameters.size())
        parm[name]=parameters.detach().numpy()
# %%
from matplotlib import pyplot as plt
import seaborn as sns
w = parm['contextiter._sub_modules_space.0.conv.weight']
# %%
sns.heatmap(w[0,0,...], annot=True)

# %%
