import torch
import torch.nn as nn
import numpy as np
from SDPA_I import SDPA_I

class multi_head_attention(nn.Module):
    def __init__(self,n_head,d_k_,d_v_,d_k,d_v,d_o):
        super().__init__()
        self.n_head=n_head
        self.d_k=d_k
        self.d_v=d_v
        self.d_o=d_o

        self.fc_q=nn.Linear(d_k_,n_head*d_k)
        self.fc_k=nn.Linear(d_k_,n_head*d_k)
        self.fc_v=nn.Linear(d_v_,n_head*d_v)
        self.attention=SDPA_I(scale=np.power(d_k,0.5))
        self.fc_o=nn.Linear(n_head*d_v,d_o)

    def forward(self,q,k,v,mask):
        d_k=self.d_k
        d_v=self.d_v
        n_head=self.n_head

        batch,n_q,d_q_=q.size()
        batch,n_k,d_k_=k.size()
        batch,n_v,d_v_=v.size()
        q=self.fc_q(q)
        k=self.fc_k(k)
        v=self.fc_v(v)

        q=q.view(batch,n_q,n_head,d_k).permute(2,0,1,3).contiguous().view(-1,n_q,d_k)
        k=k.view(batch,n_k,n_head,d_k).permute(2,0,1,3).contiguous().view(-1,n_k,d_k)
        v=v.view(batch,n_v,n_head,d_v).permute(2,0,1,3).contiguous().view(-1,n_v,d_v)

        if mask is not None:
            mask=mask.repeat(n_head,1,1)
        
        attn,output=self.attention(q,k,v,mask)
        output=output.view(n_head,batch,n_q,d_v).permute(1,2,0,3).contiguous().view(batch,n_q,-1)
        output=self.fc_o(output)

        return attn,output

if __name__=="__main__":
    n_q,n_k,n_v=2,4,4
    d_q_,d_k_,d_v_=128,128,64
    batch=10

    q=torch.randn(batch,n_q,d_q_)
    k=torch.randn(batch,n_k,d_k_)
    v=torch.randn(batch,n_v,d_v_)
    mask=torch.zeros(batch,n_q,n_k).bool()

    mha=multi_head_attention(n_head=8,d_k_=d_k_,d_v_=d_v_,d_k=256,d_v=128,d_o=128)
    attn,output=mha(q,k,v,mask=mask)
    print(attn)
    print(output)

