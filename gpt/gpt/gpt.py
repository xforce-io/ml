"""
基于 Transformer 架构的 GPT 模型实现
包含了多头注意力(MHA)、分组查询注意力(GQA)和多查询注意力(MQA)三种注意力机制
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import time
import numpy as np
from gpt.config import ModelConfig
import os
import json
from tqdm import tqdm

class AttentionBase(nn.Module):
    """注意力机制的基类，实现了基础的头部分割功能"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size  # 隐藏层维度
        self.num_heads = config.num_heads      # 注意力头数
        self.head_dim = config.hidden_size // config.num_heads  # 每个头的维度
        self.dropout = config.dropout          # dropout 比率

    def _split_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        """将输入张量分割成多个注意力头
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, dim]
            num_heads: 注意力头数量
        Returns:
            形状为 [batch_size, num_heads, seq_len, head_dim] 的张量
        """
        batch_size, seq_len, dim = x.shape
        x = x.view(batch_size, seq_len, num_heads, dim // num_heads)
        return x.permute(0, 2, 1, 3)

class MultiHeadAttention(AttentionBase):
    """标准的多头注意力实现
    每个头都有独立的 Q、K、V 投影矩阵，所有头的数量相同"""
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # 为 Q、K、V 创建独立的线性投影层
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, hidden_size]
            attention_mask: 注意力掩码，形状为 [batch_size, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # 生成 Q、K、V 并分割头部
        q = self._split_heads(self.q_proj(x), self.num_heads)  # [B, num_heads, seq_len, head_dim]
        k = self._split_heads(self.k_proj(x), self.num_heads)  # [B, num_heads, seq_len, head_dim]
        v = self._split_heads(self.v_proj(x), self.num_heads)  # [B, num_heads, seq_len, head_dim]
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim))
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, seq_len]
            scores = scores + attention_mask
        
        # 应用 softmax 和 dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # 计算输出
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        return self.out_proj(attn_output)

class GroupedQueryAttention(AttentionBase):
    """分组查询注意力实现的优化版本
    将查询头分组，每组共享相同的键值对
    优化点：
    1. 减少广播操作
    2. 优化张量重塑顺序
    3. 确保关键操作的内存连续性
    4. 支持训练和推理时使用不同的 KV heads 数量
    """
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        assert config.num_kv_heads is not None
        self.num_kv_heads = config.num_kv_heads  # 训练时的 KV 头数量
        self.inference_num_kv_heads = config.inference_num_kv_heads or config.num_kv_heads  # 推理时的 KV 头数量
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads  # 每个 KV 头对应的查询头数量
        
        # Q 保持原有维度，K、V 维度减少
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        kv_size = config.hidden_size * config.num_kv_heads // config.num_heads
        self.k_proj = nn.Linear(config.hidden_size, kv_size)
        self.v_proj = nn.Linear(config.hidden_size, kv_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        # 用于推理时重塑 K、V 的投影矩阵
        if self.inference_num_kv_heads != self.num_kv_heads:
            inference_kv_size = config.hidden_size * self.inference_num_kv_heads // config.num_heads
            self.inference_k_proj = nn.Linear(kv_size, inference_kv_size)
            self.inference_v_proj = nn.Linear(kv_size, inference_kv_size)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """优化的前向传播
        Args:
            x: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
        Returns:
            [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. 首先计算 Q、K、V 投影，确保结果是连续的
        q = self.q_proj(x).contiguous()  # [B, S, H]
        k = self.k_proj(x).contiguous()  # [B, S, kv_size]  其中 kv_size = H * num_kv_heads / num_heads
        v = self.v_proj(x).contiguous()  # [B, S, kv_size]
        
        # 如果是推理模式且 KV heads 数量不同，则重新投影 K、V
        if not self.training and self.inference_num_kv_heads != self.num_kv_heads:
            k = self.inference_k_proj(k).contiguous()  # [B, S, inference_kv_size]
            v = self.inference_v_proj(v).contiguous()  # [B, S, inference_kv_size]
            current_num_kv_heads = self.inference_num_kv_heads
            current_queries_per_kv = self.num_heads // self.inference_num_kv_heads
        else:
            current_num_kv_heads = self.num_kv_heads
            current_queries_per_kv = self.num_queries_per_kv
        
        # 2. 重塑 Q
        # 首先分成 num_heads 个头
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)  # [B, S, H/D, D]
        # 然后重组为 KV 分组形式
        q = q.view(batch_size, seq_len, current_num_kv_heads, current_queries_per_kv, self.head_dim)  # [B, S, KV_H, Q_per_KV, D]
        q = q.permute(0, 2, 3, 1, 4)  # [B, KV_H, Q_per_KV, S, D]
        
        # 3. 重塑 K 和 V
        # [B, S, kv_size] -> [B, S, num_kv_heads, head_dim]
        k = k.view(batch_size, seq_len, current_num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, current_num_kv_heads, self.head_dim)
        k = k.permute(0, 2, 1, 3)  # [B, num_kv_heads, S, head_dim]
        v = v.permute(0, 2, 1, 3)  # [B, num_kv_heads, S, head_dim]
        
        # 4. 计算注意力分数
        # [B, num_kv_heads, queries_per_kv, S, head_dim] @ [B, num_kv_heads, head_dim, S] 
        # -> [B, num_kv_heads, queries_per_kv, S, S]
        scores = torch.einsum('bhqsd,bhkd->bhqsk', q, k) / torch.sqrt(torch.tensor(self.head_dim))
        
        # 5. 处理注意力掩码
        if attention_mask is not None:
            # [B, S] -> [B, 1, 1, 1, S]
            attention_mask = attention_mask.view(batch_size, 1, 1, 1, seq_len)
            scores = scores + attention_mask
        
        # 6. 应用 softmax 和 dropout
        attn_weights = F.softmax(scores, dim=-1)  # [B, num_kv_heads, queries_per_kv, S, S]
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # 7. 计算输出
        # [B, num_kv_heads, queries_per_kv, S, S] @ [B, num_kv_heads, S, head_dim]
        # -> [B, num_kv_heads, queries_per_kv, S, head_dim]
        attn_output = torch.einsum('bhqsk,bhkd->bhqsd', attn_weights, v)
        
        # 8. 重塑回原始维度
        # [B, num_kv_heads, queries_per_kv, S, head_dim] -> [B, S, num_kv_heads, queries_per_kv, head_dim]
        attn_output = attn_output.permute(0, 3, 1, 2, 4).contiguous()
        # [B, S, num_kv_heads * queries_per_kv * head_dim] = [B, S, H]
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        return self.out_proj(attn_output)  # [B, S, H]

class TransformerBlock(nn.Module):
    """Transformer 块，包含自注意力层和前馈神经网络"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_type = config.attention_type
        
        # 根据配置选择不同的注意力机制
        if config.attention_type == 'mha':
            self.attn = MultiHeadAttention(config)
        elif config.attention_type == 'gqa':
            self.attn = GroupedQueryAttention(config)
        elif config.attention_type == 'mqa':
            self.attn = GroupedQueryAttention(config)
        
        # 前馈神经网络
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout),
        )
        # Layer Normalization 层
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播，包含残差连接"""
        x = x + self.attn(self.ln1(x), attention_mask)  # 注意力层的残差连接
        x = x + self.mlp(self.ln2(x))                   # MLP 层的残差连接
        return x

class GPT(nn.Module):
    """GPT 模型的主体实现"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 词嵌入和位置编码
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer 层
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """初始化模型权重
        使用正态分布初始化线性层和嵌入层，LayerNorm 层使用 1 和 0 初始化
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
            
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """模型前向传播
        Args:
            input_ids: 输入的词 id，形状为 [batch_size, seq_len]
            attention_mask: 注意力掩码，形状为 [batch_size, seq_len]
            labels: 目标词 id，用于计算损失，形状为 [batch_size, seq_len]
        Returns:
            包含 logits 和可选的 loss 的字典
        """
        b, t = input_ids.size()
        # 生成位置编码的索引
        pos = torch.arange(0, t, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        
        # 计算词嵌入和位置编码的和
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(pos)
        
        x = self.dropout(token_embeddings + position_embeddings)
        
        # 通过所有 Transformer 层
        for block in self.blocks:
            x = block(x, attention_mask)
            
        x = self.ln_f(x)
        logits = self.head(x)
        
        outputs = {"logits": logits}
        if labels is not None:
            # 计算语言模型的损失：预测下一个词
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), 
                                shift_labels.view(-1))
            outputs["loss"] = loss
            
        return outputs
    
    def save_pretrained(self, save_dir: str):
        """保存模型权重和配置到指定目录
        Args:
            save_dir: 保存目录的路径
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存模型权重
        torch.save(self.state_dict(), os.path.join(save_dir, "model.pt"))
        
        # 保存配置
        config_dict = {
            field: getattr(self.config, field)
            for field in self.config.__dataclass_fields__
        }
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, model_dir: str, device: str = None):
        """从保存的目录加载预训练模型
        Args:
            model_dir: 模型目录路径
            device: 设备类型（'cuda'/'cpu'/'mps'）
        Returns:
            加载好的模型实例
        """
        # 检测设备
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        # 加载配置
        with open(os.path.join(model_dir, "config.json"), "r") as f:
            config_dict = json.load(f)
        config = ModelConfig(**config_dict)
        
        # 初始化模型并加载权重
        model = cls(config)
        state_dict = torch.load(os.path.join(model_dir, "model.pt"), 
                            map_location=device)
        model.load_state_dict(state_dict)
        
        return model.to(device)

    def train_step(self, batch: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """执行单步训练
        Args:
            batch: 包含训练数据的字典
            optimizer: 优化器实例
        Returns:
            包含训练指标的字典
        """
        self.train()
        optimizer.zero_grad()
        
        outputs = self(**batch)
        loss = outputs["loss"]
        loss.backward()
        
        # 梯度裁剪
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 确保在 MPS 设备上正确获取标量值
        device = next(self.parameters()).device
        if device.type == "mps":
            return {
                "loss": loss.detach().cpu().item(),
                "perplexity": torch.exp(loss.detach().cpu()).item(),
                "grad_norm": grad_norm.cpu().item()
            }
        else:
            return {
                "loss": loss.item(),
                "perplexity": torch.exp(loss).item(),
                "grad_norm": grad_norm.item()
            }
    
    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """评估模型性能
        Args:
            dataloader: 评估数据的 DataLoader
        Returns:
            包含评估指标的字典
        """
        self.eval()
        total_loss = 0
        total_samples = 0
        start_time = time.time()
        
        with torch.no_grad():
            progress_bar = tqdm(
                dataloader,
                desc="Evaluating",
                leave=False,
                ncols=100
            )
            for batch in progress_bar:
                batch = {k: v.to(next(self.parameters()).device) for k, v in batch.items()}
                outputs = self(**batch)
                total_loss += outputs["loss"].item() * len(batch["input_ids"])
                total_samples += len(batch["input_ids"])
                
                # 更新进度条
                progress_bar.set_postfix({
                    "loss": f"{total_loss/total_samples:.4f}",
                    "ppl": f"{np.exp(total_loss/total_samples):.2f}"
                })
        
        avg_loss = total_loss / total_samples
        eval_time = time.time() - start_time
        
        return {
            "loss": avg_loss,
            "perplexity": np.exp(avg_loss),
            "eval_samples_per_second": total_samples / eval_time
        }
    
    def train_epoch(self, 
                   train_dataloader: torch.utils.data.DataLoader,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   val_dataloader: Optional[torch.utils.data.DataLoader] = None) -> Dict[str, float]:
        """训练一个完整的 epoch
        Args:
            train_dataloader: 训练数据的 DataLoader
            optimizer: 优化器实例
            epoch: 当前 epoch 编号
            val_dataloader: 可选的验证数据 DataLoader
        Returns:
            包含训练和验证指标的字典
        """
        # 如果是 DistributedDataParallel 模型，获取原始模型
        model = self.module if hasattr(self, 'module') else self
        
        epoch_start_time = time.time()
        
        running_loss = 0.0
        running_samples = 0
        running_grad_norm = 0.0
        num_steps = 0
        
        # 创建进度条
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch:3d}",
            ncols=100,
            leave=True
        )
        
        for batch in progress_bar:
            batch_start_time = time.time()
            batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}
            
            metrics = model.train_step(batch, optimizer)
            batch_size = len(batch["input_ids"])
            running_loss += metrics["loss"] * batch_size
            running_samples += batch_size
            running_grad_norm += metrics["grad_norm"]
            num_steps += 1
            
            # 更新进度条
            avg_loss = running_loss / running_samples
            avg_grad_norm = running_grad_norm / num_steps
            batch_time = time.time() - batch_start_time
            samples_per_sec = batch_size / batch_time
            
            progress_bar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "ppl": f"{np.exp(avg_loss):.2f}",
                "grad": f"{avg_grad_norm:.2f}",
                "speed": f"{samples_per_sec:.1f}s/s"
            })
        
        train_time = time.time() - epoch_start_time
        
        # 计算训练指标
        epoch_metrics = {
            "train_loss": running_loss / running_samples,
            "train_perplexity": np.exp(running_loss / running_samples),
            "train_grad_norm": running_grad_norm / num_steps,
            "train_samples_per_second": running_samples / train_time,
            "epoch_time": train_time
        }
        
        # 如果提供了验证集，进行验证
        if val_dataloader is not None:
            val_metrics = model.evaluate(val_dataloader)
            epoch_metrics.update({
                "val_loss": val_metrics["loss"],
                "val_perplexity": val_metrics["perplexity"],
                "val_samples_per_second": val_metrics["eval_samples_per_second"]
            })
        
        # 打印 epoch 总结
        if torch.distributed.get_rank() == 0:  # 只在主进程打印
            print(f"\nEpoch {epoch} Summary:")
            print(
                f"Training - Loss: {epoch_metrics['train_loss']:.4f}, "
                f"PPL: {epoch_metrics['train_perplexity']:.2f}, "
                f"Grad Norm: {epoch_metrics['train_grad_norm']:.2f}, "
                f"Speed: {epoch_metrics['train_samples_per_second']:.1f} samples/s"
            )
            if val_dataloader is not None:
                print(
                    f"Validation - Loss: {epoch_metrics['val_loss']:.4f}, "
                    f"PPL: {epoch_metrics['val_perplexity']:.2f}, "
                    f"Speed: {epoch_metrics['val_samples_per_second']:.1f} samples/s"
                )
            print(f"Time - Training: {epoch_metrics['epoch_time']:.2f}s")
            print("-" * 80)
        
        return epoch_metrics