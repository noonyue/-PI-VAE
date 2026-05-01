"""
Transformer Encoder for Spectral Data
基于Transformer的光谱编码器模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    位置编码模块
    为光谱序列添加位置信息，保留波长顺序
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制
    捕获光谱不同波段之间的长程依赖关系
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Q, K, V线性变换
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 输出线性变换
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, x, mask=None, return_attention=False):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: optional attention mask
            return_attention: 是否返回注意力权重
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, n_heads, seq_len, seq_len) if return_attention=True
        """
        batch_size, seq_len, d_model = x.size()

        # 线性变换并分割为多头
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 使用PyTorch内置的scaled_dot_product_attention（Flash Attention）
        # 显存复杂度从O(seq_len^2)降低到O(seq_len)
        if not return_attention:
            context = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0
            )
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
            return self.W_o(context)

        # 需要返回注意力权重时，回退到显式计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.W_o(context), attention_weights


class FeedForward(nn.Module):
    """
    前馈神经网络
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerEncoderBlock(nn.Module):
    """
    Transformer编码器块
    包含多头注意力 + 前馈网络 + 残差连接 + 层归一化
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None, return_attention=False):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        # 多头注意力 + 残差连接 + 层归一化
        if return_attention:
            attn_output, attention_weights = self.attention(x, mask, return_attention=True)
        else:
            attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        if return_attention:
            return x, attention_weights
        return x


class SpectralTransformerEncoder(nn.Module):
    """
    光谱Transformer编码器
    用于替代原始PI-VAE中的MLP编码器

    输入: (batch, spectrum_dim) 光谱数据
    输出: (batch, latent_dim) 潜变量 (mu, logvar)
    """
    def __init__(self, spectrum_dim, latent_dim, d_model=256, n_heads=8, n_layers=4,
                 d_ff=512, dropout=0.1):
        super().__init__()
        self.spectrum_dim = spectrum_dim
        self.latent_dim = latent_dim
        self.d_model = d_model

        # 输入投影：将光谱维度映射到d_model
        self.input_projection = nn.Linear(1, d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len=spectrum_dim, dropout=dropout)

        # Transformer编码器层
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # 输出层：映射到潜变量
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)

    def forward(self, x, return_attention=False):
        """
        Args:
            x: (batch, spectrum_dim) 光谱数据
            return_attention: 是否返回注意力权重
        Returns:
            mu: (batch, latent_dim)
            logvar: (batch, latent_dim)
            attention_weights: list of (batch, n_heads, seq_len, seq_len) if return_attention=True
        """
        batch_size = x.size(0)

        # 重塑为序列格式: (batch, spectrum_dim) -> (batch, spectrum_dim, 1)
        x = x.unsqueeze(-1)

        # 输入投影: (batch, spectrum_dim, 1) -> (batch, spectrum_dim, d_model)
        x = self.input_projection(x)

        # 位置编码
        x = self.pos_encoding(x)

        # 通过Transformer编码器层
        attention_weights_list = []
        for layer in self.encoder_layers:
            if return_attention:
                x, attn_weights = layer(x, return_attention=True)
                attention_weights_list.append(attn_weights)
            else:
                x = layer(x)

        # 全局平均池化: (batch, spectrum_dim, d_model) -> (batch, d_model)
        x = x.mean(dim=1)

        # 映射到潜变量
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        if return_attention:
            return mu, logvar, attention_weights_list
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        重参数化技巧
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


# 测试代码
if __name__ == "__main__":
    # 测试Transformer编码器
    batch_size = 16
    spectrum_dim = 200  # UV-Vis光谱维度
    latent_dim = 32

    # 创建模型
    encoder = SpectralTransformerEncoder(
        spectrum_dim=spectrum_dim,
        latent_dim=latent_dim,
        d_model=256,
        n_heads=8,
        n_layers=4,
        d_ff=512,
        dropout=0.1
    )

    # 创建随机输入
    x = torch.randn(batch_size, spectrum_dim)

    # 前向传播
    mu, logvar = encoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")

    # 测试注意力权重提取
    mu, logvar, attention_weights = encoder(x, return_attention=True)
    print(f"\nNumber of attention weight tensors: {len(attention_weights)}")
    print(f"Attention weights shape (layer 0): {attention_weights[0].shape}")

    # 测试重参数化
    z = encoder.reparameterize(mu, logvar)
    print(f"Latent variable z shape: {z.shape}")

    print("\n✓ Transformer编码器测试通过！")
