# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#%%
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F


def find_named_module(module, query):
    """Helper function to find a named module. Returns a `nn.Module` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the module name to find

    Returns:
        nn.Module or None
    """

    return next((m for n, m in module.named_modules() if n == query), None)


def find_named_buffer(module, query):
    """Helper function to find a named buffer. Returns a `torch.Tensor` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the buffer name to find

    Returns:
        torch.Tensor or None
    """
    return next((b for n, b in module.named_buffers() if n == query), None)


def _update_registered_buffer(
    module,
    buffer_name,
    state_dict_key,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    new_size = state_dict[state_dict_key].size()
    registered_buf = find_named_buffer(module, buffer_name)

    if policy in ("resize_if_empty", "resize"):
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')

        if policy == "resize" or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)

    elif policy == "register":
        if registered_buf is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')

        module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))

    else:
        raise ValueError(f'Invalid policy "{policy}"')


def update_registered_buffers(
    module,
    module_name,
    buffer_names,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    """Update the registered buffers in a module according to the tensors sized
    in a state_dict.

    (There's no way in torch to directly load a buffer with a dynamic size)

    Args:
        module (nn.Module): the module
        module_name (str): module name in the state dict
        buffer_names (list(str)): list of the buffer names to resize in the module
        state_dict (dict): the state dict
        policy (str): Update policy, choose from
            ('resize_if_empty', 'resize', 'register')
        dtype (dtype): Type of buffer to be registered (when policy is 'register')
    """
    valid_buffer_names = [n for n, _ in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')

    for buffer_name in buffer_names:
        _update_registered_buffer(
            module,
            buffer_name,
            f"{module_name}.{buffer_name}",
            state_dict,
            policy,
            dtype,
        )


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )


def quantize_ste(x):
    """Differentiable quantization via the Straight-Through-Estimator."""
    # STE (straight-through estimator) trick: x_hard - x_soft.detach() + x_soft
    return (torch.round(x) - x).detach() + x


def gaussian_kernel1d(
    kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype
):
    """1D Gaussian kernel."""
    khalf = (kernel_size - 1) / 2.0
    x = torch.linspace(-khalf, khalf, steps=kernel_size, dtype=dtype, device=device)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    return pdf / pdf.sum()


def gaussian_kernel2d(
    kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype
):
    """2D Gaussian kernel."""
    kernel = gaussian_kernel1d(kernel_size, sigma, device, dtype)
    return torch.mm(kernel[:, None], kernel[None, :])


def gaussian_blur(x, kernel=None, kernel_size=None, sigma=None):
    """Apply a 2D gaussian blur on a given image tensor."""
    if kernel is None:
        if kernel_size is None or sigma is None:
            raise RuntimeError("Missing kernel_size or sigma parameters")
        dtype = x.dtype if torch.is_floating_point(x) else torch.float32
        device = x.device
        kernel = gaussian_kernel2d(kernel_size, sigma, device, dtype)

    padding = kernel.size(0) // 2
    x = F.pad(x, (padding, padding, padding, padding), mode="replicate")
    x = torch.nn.functional.conv2d(
        x,
        kernel.expand(x.size(1), 1, kernel.size(0), kernel.size(1)),
        groups=x.size(1),
    )
    return x


def meshgrid2d(N: int, C: int, H: int, W: int, device: torch.device):
    """Create a 2D meshgrid for interpolation."""
    theta = torch.eye(2, 3, device=device).unsqueeze(0).expand(N, 2, 3)
    return F.affine_grid(theta, (N, C, H, W), align_corners=False)


class Space2Depth(nn.Module):
    """
    ref: https://github.com/huzi96/Coarse2Fine-PyTorch/blob/master/networks.py
    """

    def __init__(self, r=2):
        super().__init__()
        self.r = r

    def forward(self, x):
        r = self.r
        b, c, h, w = x.size()
        out_c = c * (r**2)
        out_h = h // r
        out_w = w // r
        x_view = x.view(b, c, out_h, r, out_w, r)
        x_prime = x_view.permute(0, 3, 5, 1, 2, 4).contiguous().view(b, out_c, out_h, out_w)
        return x_prime


class Depth2Space(nn.Module):
    def __init__(self, r=2):
        super().__init__()
        self.r = r

    def forward(self, x):
        r = self.r
        b, c, h, w = x.size()
        out_c = c // (r**2)
        out_h = h * r
        out_w = w * r
        x_view = x.view(b, r, r, out_c, h, w)
        x_prime = x_view.permute(0, 3, 4, 1, 5, 2).contiguous().view(b, out_c, out_h, out_w)
        return x_prime


def Demultiplexer(x):
    """
    See Supplementary Material: Figure 2.
    This operation can also implemented by slicing.
    """
    x_prime = Space2Depth(r=2)(x)
    
    _, C, _, _ = x_prime.shape
    y1_index = tuple(range(0, C // 4))
    y2_index = tuple(range(C * 3 // 4, C))
    y3_index = tuple(range(C // 4, C // 2))
    y4_index = tuple(range(C // 2, C * 3 // 4))

    y1 = x_prime[:, y1_index, :, :]
    y2 = x_prime[:, y2_index, :, :]
    y3 = x_prime[:, y3_index, :, :]
    y4 = x_prime[:, y4_index, :, :]

    return y1, y2, y3, y4


def Multiplexer(y1, y2, y3, y4):
    """
    The inverse opperation of Demultiplexer.
    This operation can also implemented by slicing.
    """
    x_prime = torch.cat((y1, y3, y4, y2), dim=1)
    return Depth2Space(r=2)(x_prime)

def CheckerboardDemux(x):
    """checkerboard demutiplexer

    Args:
        x:tensor

    Returns:
        y1: anchor, y2: non_anchor
    """    
    x_prime = Space2Depth(r=2)(x)
    
    _, C, _, _ = x_prime.shape
    y1_index = tuple(range(0, C // 4)) + tuple(range(C * 3 // 4, C))
    y2_index = tuple(range(C // 4, C * 3 // 4))
    
    y1 = torch.index_select(x_prime, 1, torch.as_tensor(y1_index, device=x.device))
    y2 = torch.index_select(x_prime, 1, torch.as_tensor(y2_index, device=x.device))
    return y1, y2

def CheckerboardMux(y1, y2):
    C = y1.shape[1]
    x_prime = torch.cat([y1[:, :C // 2, ...], y2, y1[:, C // 2:, ...]], dim=1)
    return Depth2Space(r=2)(x_prime)



class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False, padding_mode="replicate")

        Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]])
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x


def gumbelSoftmax(logits: torch.Tensor, temperature: float = 1.0, hard: bool = True, dim: int = -1, elements=1):
    eps = torch.finfo(logits.dtype).eps
    uniforms = torch.rand_like(logits).clamp_(eps, 1 - eps)
    gumbels = -((-(uniforms.log())).log())

    logits = (logits + gumbels) / temperature
    # y_soft = torch.sigmoid(logits)
    # logits = F.normalize(logits, p=2, dim=dim)
    # y_soft = torch.exp(logits - 1)
    # y_soft = F.normalize(y_soft, p=float('inf'), dim=dim)
    y_soft = torch.exp(logits - logits.max(dim)[0].unsqueeze(dim))
    # y_soft = ((logits + gumbels) / temperature).softmax(dim)

    if hard:
        # Straight through.
        index = torch.topk(y_soft, elements, dim, largest=True, sorted=False)[1]
        # index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret



def sinkhorn_forward(C,mu,nu,epsilon,max_iter):
    bs,n,k_=C.size()
    v = torch.ones([bs,1,k_])/(k_)
    G = torch.exp(-C / epsilon)
    
    if torch.cuda.is_available():
        v =v.cuda()

    for i in range(max_iter):
        u=mu/(G*v).sum(-1,keepdim=True)
        v=nu/(G*u).sum(-2,keepdim=True)
    
    Gamma=u*G*v
    return Gamma


def sinkhorn_forward_stablized(C,mu,nu,epsilon,max_iter):
    bs,n,k_=C.size()
    k=k_-1
    
    f=torch.zeros([bs,n,1])
    g=torch.zeros([bs,1,k+1])
    if torch.cuda.is_available():
        f=f.cuda()
        g=g.cuda()
    
    epsilon_log_mu=epsilon*torch.log(mu)
    epsilon_log_nu=epsilon*torch.log(nu)
        
    def min_epsilon_row(Z,epsilon):
        return -epsilon*torch.logsumexp((-Z)/epsilon,-1,keepdim=True)
    
    def min_epsilon_col(Z,epsilon):
        return -epsilon*torch.logsumexp((-Z)/epsilon,-2,keepdim=True)
    
    for i in range(max_iter):
        f = min_epsilon_row(C-g, epsilon) + epsilon_log_mu
        g = min_epsilon_col(C-f, epsilon) + epsilon_log_nu
        
    Gamma = torch.exp((-C+f+g)/epsilon)
    return Gamma
    
def sinkhorn_backward(grad_output_Gamma,Gamma,mu,nu,epsilon):
    nu_=nu[:,:,:-1]
    Gamma_=Gamma[:,:,:-1]
    
    bs,n,k_=Gamma.size()
    
    inv_mu = 1./(mu.view([1, -1])) #[1,n]
    Kappa=torch.diag_embed(nu_.squeeze(-2)) \
        -torch.matmul(Gamma_.transpose(-1,-2) * inv_mu.unsqueeze(-2),Gamma_) #[bs,k,k]
    
    inv_Kappa = torch.inverse(Kappa) #[bs,k,k]
    
    Gamma_mu = inv_mu.unsqueeze(-1)*Gamma_
    
    L = Gamma_mu.matmul(inv_Kappa) #[bs,n,k]
    
    G1 = grad_output_Gamma * Gamma #[bs,n,k+1]
    
    g1=G1.sum(-1)
    G21=(g1*inv_mu).unsqueeze(-1)*Gamma #[bs,n,k+1]
    g1_L=g1.unsqueeze(-2).matmul(L) #[bs,1,k]
    G22=g1_L.matmul(Gamma_mu.transpose(-1,-2)).transpose(-1,-2)*Gamma #[bs,n,k+1]
    G23 = - F.pad(g1_L,pad=(0,1),mode='constant',value=0)*Gamma #[bs,n,k+1]
    G2=G21+G22+G23 #[bs,n,k+1]
    
    del g1,G21,G22,G23,Gamma_mu
    
    g2=G1.sum(-2).unsqueeze(-1) #[bs,k+1,1]
    g2=g2[:,:-1,:] #[bs,k,1]
    G31 = -L.matmul(g2)*Gamma #[bs,n,k+1]
    G32=F.pad(inv_Kappa.matmul(g2).transpose(-1,-2),pad=(0,1),mode='constant',value=0)*Gamma #[bs,n,k+1]
    G3=G31+G32 #[bs,n,k+1]
    grad_C = (-G1+G2+G3)/epsilon #[bs,n,k+1]
    return grad_C

class TopKFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C, mu, nu, epsilon, max_iter):
        with torch.no_grad():
            if epsilon>1e-2:
                Gamma=sinkhorn_forward(C,mu,nu,epsilon,max_iter)
                
                if bool(torch.any(Gamma!=Gamma)):
                    print('Nan appeared in Gamma,re-computing...')
                    Gamma=sinkhorn_forward_stablized(C,mu,nu,epsilon,max_iter)
            else:
                Gamma=sinkhorn_forward_stablized(C,mu,nu,epsilon,max_iter)
            
            ctx.save_for_backward(mu,nu,Gamma)
            ctx.epsilon=epsilon
        return Gamma
        
    @staticmethod
    def backward(ctx,grad_output_Gamma):
        epsilon=ctx.epsilon
        mu,nu,Gamma=ctx.saved_tensors 
        #mu[1,n,1]
        # #nu[1,1,k+1]
        # #Gamma[bs,n,k+1]
        with torch.no_grad():
            grad_C=sinkhorn_backward(grad_output_Gamma,Gamma,mu,nu,epsilon)
            return grad_C,None,None,None,None

class TopK_custom(torch.nn.Module):
    def __init__(self,k,epsilon=0.1,max_iter=200):
        super(TopK_custom,self).__init__()
        self.k=k
        self.epsilon=epsilon
        self.anchors=torch.FloatTensor([k-i for i in range(k+1)]).view([1,1,k+1])
        self.max_iter=max_iter
        if torch.cuda.is_available():
            self.anchors=self.anchors.cuda()
    
    def setK(self, new_k):
        if new_k == self.k:
            return
        self.k = new_k
        self.anchors=torch.FloatTensor([new_k-i for i in range(new_k+1)]).view([1,1,new_k+1])
        if torch.cuda.is_available():
            self.anchors=self.anchors.cuda()
    
    def forward(self,scores):
        bs,n=scores.size()
        scores=scores.view([bs,n,1]) 
        #find the -inf value and replace it with the minimum value except -inf 
        scores_=scores.clone().detach()
        max_scores=torch.max(scores_).detach()
        scores_[scores_==float('-inf')]=float('inf')
        min_scores=torch.min(scores_).detach()
        filled_value=min_scores-(max_scores-min_scores)
        mask=scores==float('-inf')
        scores=scores.masked_fill(mask,filled_value)
        
        C=(scores-self.anchors)**2
        C=C/(C.max().detach())
        
        mu=torch.ones([1,n,1],requires_grad=False)/n
        nu=[1./n for _ in range(self.k)]
        nu.append((n-self.k)/n)
        nu=torch.FloatTensor(nu).view([1,1,self.k+1])
        if torch.cuda.is_available():
            mu=mu.cuda()
            nu=nu.cuda()
        
        Gamma=TopKFunc.apply(C,mu,nu,self.epsilon,self.max_iter)
        A=Gamma[:,:,:]*n 
        return A,None


class gumbelSoftTopk(nn.Module):
    def __init__(self, k=10) -> None:
        super().__init__()
        self.k = k
        self.topk = TopK_custom(self.k, epsilon=1e-3, max_iter=1500)
    
    def setK(self, new_k):
        self.topk.setK(new_k)
    
    def forward(self, logits: torch.Tensor, temperature: float = 1.0, hard: bool = True, dim: int = -1, elements=1):
        eps = torch.finfo(logits.dtype).eps
        uniforms = torch.rand_like(logits).clamp_(eps, 1 - eps)
        gumbels = -((-(uniforms.log())).log())

        logits = (logits + gumbels) / temperature
        self.setK(elements)
        A_topk = self.topk(logits.squeeze(1))[0]
        A_topk = (1 - A_topk[..., -1].unsqueeze(1))
        y_soft = (logits + torch.log(A_topk)).softmax(dim)

        if hard:
            # Straight through.
            index = torch.topk(y_soft, elements, dim, largest=True, sorted=False)[1]
            # index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret


if __name__ == "__main__":
    
    x = torch.randint(0, 2*64*32*32, (2,64,32,32))
    x[..., 0::2, 1::2] = 0
    x[..., 1::2, 0::2] = 0
    y1, y2 = CheckerboardDemux(x)
    x_hat = CheckerboardMux(y1, y2)
    print((x == x_hat).all())
#%%