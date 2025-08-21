import math
import torch
import torch.nn as nn

class DownsampleLY(nn.Module):
    def __init__(self, learnable=False):
        super(DownsampleLY, self).__init__()
        
        a = torch.tensor([[0.0062, 0.000, -0.0062]], dtype=torch.float32) 
        b = torch.tensor([[1.000, -1.9866, 0.9875]], dtype=torch.float32)  

        A = torch.fft.fft(a, n=512)  
        B = torch.fft.fft(b, n=512)  

        H = A/B
        self.register_buffer('H', torch.abs(H).to(torch.float32))
        
        fir_weights = torch.tensor([[[0.25, 0.5, 0.25]]], dtype=torch.float32)
        
        self.filter1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.filter1.weight = nn.Parameter(fir_weights, requires_grad=learnable)
        
        self.filter2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.filter2.weight = nn.Parameter(fir_weights.clone(), requires_grad=learnable)
        
    	
        
    def forward(self, x):
        X = torch.fft.fft(x, n=512)
        X = X * ((self.H)**2) 

        x = torch.fft.ifft(X, n=512).real
        
        x = x.unsqueeze(1) 
        
        x = self.filter1(x)   
        x =  x[:, :, ::2]    
        x = self.filter2(x)   
        x =  x[:, :, ::2]    
        
        x = x.squeeze(1)    
        
        return x

        
def hardthreshold(x,t):
    x = torch.mul(torch.sign(x), torch.nn.functional.relu(torch.abs(x)-torch.abs(t)))
    y=x+torch.sign(x)*torch.abs(t)
    
    return y
        
        
def create_dct_ii_matrix(N):
    dct_mat = torch.zeros(N, N)
    alpha = torch.sqrt(torch.tensor([1/N] + [2/N] * (N-1)))
    
    for k in range(N):
        for n in range(N):
            dct_mat[k][n] = alpha[k] * math.cos(math.pi * k * (2 * n + 1) / (2 * N))
    return dct_mat


def discrete_cosine_transform(u, axis=-1):
    if axis != -1:
        u = u.transpose(-1, axis)

    n = u.shape[-1]
    D = create_dct_ii_matrix(n).to(dtype=u.dtype, device=u.device)
    y = torch.matmul(u, D)  
    
    if axis != -1:
        y = y.transpose(-1, axis)
        
    return y


class DCT1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.v = torch.nn.Parameter(torch.rand(16))
        self.T = torch.nn.Parameter(torch.ones(16)*0.1)
    def forward(self, x):
        # Apply the discrete cosine transform directly
        dct_coeffs = discrete_cosine_transform(x)
        dct_coeffs = self.v*hardthreshold(dct_coeffs, self.T)
        
        return dct_coeffs


   
class DCTNN(nn.Module):
    def __init__(self):
        super(DCTNN, self).__init__()
        # Integrate the custom Downsample layer
        self.DownsampleLayer = DownsampleLY(learnable=True)
        self.encoder = nn.Sequential(
            nn.Linear(128, 16),
            nn.Linear(16, 16),
            nn.ReLU(),
        )
        self.dct_layer = DCT1D()
        self.decoder = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        x = self.DownsampleLayer(x)
        x = self.encoder(x)
        x = self.dct_layer(x)
        x = self.decoder(x)
        return x
    
    
