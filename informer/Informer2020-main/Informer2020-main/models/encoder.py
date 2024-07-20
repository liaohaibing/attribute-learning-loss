import torch
import torch.nn as nn
import torch.nn.functional as F


"""ConvLayer类实现的是Informer中的Distilling操作，本质上就是个1维卷积+ELU激活函数+
   最大池化"""
class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1,2)
        return x


"""EncoderLayer类实现的是一个Encoder层，整体架构和Transformer是大致相同的，主要包含
   两个子层：多头注意力层（Informer中改为提出的ProbSpase Self-Attention层）和两个
   线性映射的前馈层（Feed Forward），两个子层后都带有一个批量归一化层，子层之间有跳跃连接"""
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model) #归一化方法
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn = self.attention( # new_x 注意力分数 32 96 512
            x, x, x,
            attn_mask = attn_mask
        )
        #残差连接 加快收敛速度 防止梯度消失和梯度爆炸 无法更新和超出最大值溢出
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm2(x+y), attn  #归一化输出


"""Encoder类是将前面定义的Encoder层和Distilling操作组织起来，形成一个Encoder模块。
   其中distilling层总比EncoderLayer少一层，即最后一层EncoderLayer后不再做distilling操作"""
class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask) #x 注意力分数
                x = conv_layer(x) #降维 32 96 512 变为 32 48 512
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)  #32 48 512
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


"""为了增强distilling操作的鲁棒性，文章中提到可以采用多个replicas并行执行，
   不同长度的embedding，embedding长度减半对应的attention层也减少一层，
   distilling层也会随之减少一层，最终得到的结果拼接起来输出"""
class EncoderStack(nn.Module):
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        x_stack = []; attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1]//(2**i_len)
            x_s, attn = encoder(x[:, -inp_len:, :])
            x_stack.append(x_s); attns.append(attn)
        x_stack = torch.cat(x_stack, -2)
        
        return x_stack, attns
