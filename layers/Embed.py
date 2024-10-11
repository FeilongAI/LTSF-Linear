import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):


    def __init__(self, d_model, max_len=5000):#模型的维度 最大序列长度，默认为 5000
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        # 创建一个大小为 (max_len, d_model) 的零张量，并设置不需要计算梯度。
        pe = torch.zeros(max_len, d_model).float()#torch.Size([5000, 512])
        pe.require_grad = False
        #创建一个从 0 到 max_len-1 的位置向量，并增加一个维度。 (max_len, 1)
        position = torch.arange(0, max_len).float().unsqueeze(1) #torch.Size([5000])->torch.Size([5000, 1])
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp() #torch.arange(0, d_model, 2) -> 乘以某个数值torch.Size([256])
        #pe shape after sin: torch.Size([1000, 512])
        pe[:, 0::2] = torch.sin(position * div_term)#广播position[5000,1] 到 [5000,256]
        pe[:, 1::2] = torch.cos(position * div_term)
        #pe shape after unsqueeze: torch.Size([1, 1000, 512]) 在添加一个0维度
        pe = pe.unsqueeze(0)#torch.Size([5000, 512]) -> torch.Size([1, 5000, 512]) unsqueeze(0) 在第0为维度添加
        self.register_buffer('pe', pe)#通过将其注册为缓冲区

    def forward(self, x):#x:torch.Size([32, 96, 7])
        return self.pe[:, :x.size(1)] #torch.Size([1, 96, 512])


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):#输入通道数（特征数） 模型的维度（输出通道数）
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):#x输入torch.Size([32, 96, 7])
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)# x.permute(0, 2, 1)：torch.Size([32, 7, 96]) self.tokenConv(x.permute(0, 2, 1))：torch.Size([32, 512, 96])  return torch.Size([32, 96, 512])
        return x #torch.Size([32, 96, 512])


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):#定义了一个时间特征嵌入类，仅使用一个线性层来嵌入时间特征。
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()
        # > *Q - [month]
        # > *M - [month]
        # > *W - [Day of month, week of year]
        # > *D - [Day of week, day of month, day of year]
        # > *B - [Day of week, day of month, day of year]
        # > *H - [Hour of day, day of week, day of month, day of year]
        # > *T - [Minute of hour *, hour of day, day of week, day of month, day of year]
        # > *S - [Second of minute, minute of hour, hour of day, day of week, day of month, day of year]

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}#不同的时间频率对应不同的特征
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False) #

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):#c_in输入特征的数量 d_model: 模型的维度 embed_type: 嵌入类型，默认为 'fixed'  时间频率，默认为 'h'（小时） dropout 率，默认为 0.1
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):#x:torch.Size([32, 96, 7])x_mark()时间列:torch.Size([32, 96, 4])
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x) # self.value_embedding(x):torch.Size([32, 96, 512]) torch.Size([32, 96, 512]) #torch.Size([1, 96, 512])
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)

class DataEmbedding_wo_pos_temp(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos_temp, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x)
        return self.dropout(x)

class DataEmbedding_wo_temp(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_temp, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)