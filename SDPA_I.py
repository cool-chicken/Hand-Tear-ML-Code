import torch
import numpy as np
import torch.nn as nn

class SDPA_I(nn.Module):
    def __init__(self,scale):
        super().__init__()

        self.scale=scale
        self.softmax=nn.Softmax(dim=2)

    def forward(self, q,k,v,mask):
        u=torch.bmm(q,k.transpose(1,2))
        u=u/self.scale

        if mask is not None:
            u=u.masked_fill(mask,-np.inf)

        attn=self.softmax(u)
        output=torch.bmm(attn,v)

        return attn,output

if __name__ =="__main__":
    n_q,n_k,n_v=4,4,4
    d_q,d_k,d_v=128,128,64
    batch=10

    q=torch.randn(batch,n_q,d_q)
    k=torch.randn(batch,n_k,d_k)
    v=torch.randn(batch,n_v,d_v)
    mask=torch.zeros(batch,n_q,n_k).bool()

    attention=SDPA_I(scale=np.power(d_k,0.5))
    attn,output=attention(q,k,v,mask=mask)
    print(attn)
    print(output)





