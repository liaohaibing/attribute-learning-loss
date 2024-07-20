"""mask机制用在Decoder的第一个attention层中，目的是为了保证t时刻解码的输出只依赖与t时刻之前的输出"""
"""生成的mask矩阵右上角部分为1（不包括对角线），将mask矩阵用到score矩阵上会使得mask矩阵中为1的位置在
   score矩阵中为-无穷，这样softmax之后就为0"""
"""TriangularCausalMask是用在Fullattention层上的
   ProbMask是用在ProbSparseAttention层上的
   """

import torch

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self):
        return self._mask