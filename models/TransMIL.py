from matplotlib.pyplot import axes, axis
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention
import random
attention = True
class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        _, attn = self.attn(self.norm(x),return_attn=True)
        x = x + _
        return x, attn



class GloalAtt(nn.Module):
    def __init__(self, dim=512, drop_rate=0.):
        super(GloalAtt, self).__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.act = nn.GELU()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.k1 = nn.Linear(dim, dim)
        self.v1 = nn.Linear(dim, dim)

        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, dilation=3, padding=3, groups=dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.k2 = nn.Linear(dim, dim)
        self.v2 = nn.Linear(dim, dim)

        self.conv3 = nn.Conv2d(dim, dim, kernel_size=3, dilation=6, padding=6, groups=dim)
        self.norm3 = nn.BatchNorm2d(dim)
        self.k3 = nn.Linear(dim, dim)
        self.v3 = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(drop_rate)

        self.proj_out = nn.Linear(dim, dim)
        # self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, dilation=1, groups=dim)
        # self.proj1 = nn.Conv2d(dim, dim, 7, 1, (7+6*2)//2, dilation=3, groups=dim)
        # self.proj2 = nn.Conv2d(dim, dim, 7, 1, (7+6*4)//2, dilation=5, groups=dim)
    def forward(self, x, H, W):
        B, N, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        q = self.proj_q(feat_token)

        kv1 = self.conv1(feat_token.permute(0, 2, 1).reshape(B, C, H ,W))
        kv1 = self.act(self.norm1(kv1).reshape(B, C, -1))
        k1 = self.k1(kv1.permute(0, 2, 1))
        v1 = self.v1(kv1.permute(0, 2, 1))
        att1 = q * k1
        # att1 = nn.Softmax()(q * k1)*v1
        att1 = nn.Softmax(dim=-1)(att1.permute(0, 2, 1))
        att1 = att1.permute(0, 2, 1)*v1

        # kv2 = self.conv2(feat_token.permute(0, 2, 1).reshape(B, C, H ,W))
        # kv2 = self.act(self.norm2(kv2).reshape(B, C, -1))
        # k2 = self.k2(kv2.permute(0, 2, 1))
        # v2 = self.v2(kv2.permute(0, 2, 1))
        # att2 = nn.Softmax()(q * k2)*v2

        # kv3 = self.conv3(feat_token.permute(0, 2, 1).reshape(B, C, H ,W))
        # kv3 = self.act(self.norm3(kv3).reshape(B, C, -1))
        # k3 = self.k3(kv3.permute(0, 2, 1))
        # v3 = self.v2(kv3.permute(0, 2, 1))
        # att3 = nn.Softmax()(q * k3)*v3

        att = att1 #+ att2 + att3
        x = torch.cat((cls_token.unsqueeze(1), att), dim=1)
        return x

class LocalAtt(nn.Module):
    def __init__(self, dim=512):
        super(LocalAtt, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, dilation=1, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 7, 1, (7+6*2)//2, dilation=3, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 7, 1, (7+6*4)//2, dilation=5, groups=dim)
    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, n_classes):
        super(TransMIL, self).__init__()
        self.local_att = LocalAtt(dim=512)
        self.global_att = GloalAtt(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)


    def forward(self, **kwargs):
        h = kwargs['data'].float() #[B, n, 1024]
        
        h = self._fc1(h) #[B, n, 512]
        
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        index = torch.randint(0,H,(add_length,1)).squeeze(-1)
        # h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]
        h = torch.cat([h, h[:,index,:]],dim = 1)
        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h, attn0 = self.layer1(h) #[B, N, 512]

        #---->PPEG
        local_att = self.local_att(h, _H, _W) #[B, N, 512]
        global_att = self.global_att(h, _H, _W)
        # ---->Translayer x2
        h, attn1 = self.layer2(local_att + global_att) #[B, N, 512]
        # h = self.layer2(local_att )
 
        #---->cls_token
        h = self.norm(h)
        # att = F.softmax(h, dim = 2)
        # h = h[:,0]

        #---->predict
        logits = self._fc2(h[:,0]) #[B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        if attention:
            attn1 = attn1[:,:H+1].to('cpu')
            attn0 = attn0[:,:H+1].to('cpu')
            result = torch.ones(attn1.shape).to(attn1.device)
            # for i, attn in enumerate([attn0, attn1]):
            for i, attn in enumerate([attn1]):
                attn = attn / (attn.max())
                result = ((attn * result) + result) / 2
            attns = result[0, 1:].to('cpu')
            # if int(Y_hat) == 1:
            epsilon = 1e-10
            attns = attns + epsilon
            attns = attns.exp()
            min_val = attns.min()
            max_val = attns.max()
            attns = (attns - min_val) / (max_val - min_val)
        else:
            attns = h[1:H]
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 
                        'Y_hat': Y_hat, 'attention':attns}
        return results_dict



if __name__ == "__main__":
    data = torch.randn((1, 6000, 1024)).cuda()
    model = TransMIL(n_classes=2).cuda()
    print(model.eval())
    results_dict = model(data = data)
    print(results_dict)
