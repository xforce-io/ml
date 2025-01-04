import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict
import time
import psutil
import numpy as np
from torch.cuda import max_memory_allocated, reset_peak_memory_stats
from gpt.config import ModelConfig
import os
import json
from datasets import load_dataset
from tqdm import tqdm

class AttentionBase(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.dropout = config.dropout

    def _split_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        x = x.view(batch_size, seq_len, num_heads, dim // num_heads)
        return x.permute(0, 2, 1, 3)  # (batch, heads, seq_len, head_dim)

class MultiHeadAttention(AttentionBase):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self._split_heads(self.q_proj(x), self.num_heads)
        k = self._split_heads(self.k_proj(x), self.num_heads)
        v = self._split_heads(self.v_proj(x), self.num_heads)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim))
        if attention_mask is not None:
            # 调整 attention_mask 的维度以匹配 scores
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, seq_len]
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        return self.out_proj(attn_output)

class GroupedQueryAttention(AttentionBase):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        assert config.num_kv_heads is not None
        self.num_kv_heads = config.num_kv_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size * config.num_kv_heads // config.num_heads)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size * config.num_kv_heads // config.num_heads)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self._split_heads(self.q_proj(x), self.num_heads)  # [B, num_heads, seq_len, head_dim]
        k = self._split_heads(self.k_proj(x), self.num_kv_heads)  # [B, num_kv_heads, seq_len, head_dim]
        v = self._split_heads(self.v_proj(x), self.num_kv_heads)  # [B, num_kv_heads, seq_len, head_dim]
        
        # 重塑 q 以匹配 k,v 的分组
        # [B, num_heads, seq_len, head_dim] -> [B, num_kv_heads, num_heads_per_kv, seq_len, head_dim]
        q = q.view(batch_size, self.num_kv_heads, self.num_heads // self.num_kv_heads, seq_len, self.head_dim)
        
        # 计算注意力分数，现在 k 无需复制
        # [B, num_kv_heads, num_heads_per_kv, seq_len, seq_len]
        scores = torch.matmul(q, k.unsqueeze(2).transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim))
        
        if attention_mask is not None:
            # 调整 attention_mask 的维度以匹配 scores
            # scores shape: [B, num_kv_heads, num_heads_per_kv, seq_len, seq_len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 1, seq_len]
            scores = scores + attention_mask
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # 计算输出，v 也无需复制
        # [B, num_kv_heads, num_heads_per_kv, seq_len, head_dim]
        attn_output = torch.matmul(attn_weights, v.unsqueeze(2))
        
        # 重塑回原始维度
        # [B, num_heads, seq_len, head_dim]
        attn_output = attn_output.view(batch_size, self.num_heads, seq_len, self.head_dim)
        
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        return self.out_proj(attn_output)

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_type = config.attention_type
        
        if config.attention_type == 'mha':
            self.attn = MultiHeadAttention(config)
        elif config.attention_type == 'gqa':
            self.attn = GroupedQueryAttention(config)
        elif config.attention_type == 'mqa':
            config.num_kv_heads = 1
            self.attn = GroupedQueryAttention(config)
        
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout),
        )
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
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
        b, t = input_ids.size()
        pos = torch.arange(0, t, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(pos)
        
        x = self.dropout(token_embeddings + position_embeddings)
        
        for block in self.blocks:
            x = block(x, attention_mask)
            
        x = self.ln_f(x)
        logits = self.head(x)
        
        outputs = {"logits": logits}
        if labels is not None:
            # 计算损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), 
                                shift_labels.view(-1))
            outputs["loss"] = loss
            
        return outputs
    
    def save_pretrained(self, save_dir: str):
        """保存模型和配置"""
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
        """从保存的目录加载模型"""
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
        
        # 初始化模型
        model = cls(config)
        
        # 加载权重
        state_dict = torch.load(os.path.join(model_dir, "model.pt"), 
                            map_location=device)
        model.load_state_dict(state_dict)
        
        return model.to(device)

    def train_step(self, batch: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """单步训练"""
        self.train()
        optimizer.zero_grad()
        
        outputs = self(**batch)
        loss = outputs["loss"]
        loss.backward()
        
        # 记录梯度范数
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
        """评估模型"""
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
        """训练一个 epoch"""
        from tqdm import tqdm
        
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
            batch = {k: v.to(next(self.parameters()).device) for k, v in batch.items()}
            
            metrics = self.train_step(batch, optimizer)
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
            val_metrics = self.evaluate(val_dataloader)
            epoch_metrics.update({
                "val_loss": val_metrics["loss"],
                "val_perplexity": val_metrics["perplexity"],
                "val_samples_per_second": val_metrics["eval_samples_per_second"]
            })
        
        # 打印 epoch 总结
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