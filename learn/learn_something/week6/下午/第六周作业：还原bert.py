import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel
import math

bert = BertModel.from_pretrained(r"D:\学习\从头学py\week6\bert_base_chinese", return_dict=False)
state_dict = bert.state_dict()

# 注意layer_norm的w是h，b也是h，所以总参数是为2和。

# self.embedding_layer_norm = nn.LayerNorm(768)  # 默认在最后一维 axis=-1 但实际应该在 axis=1 做归一化

# 问题：BERT 官方权重的 key 名字写错了。self.embedding_layer_norm(weights['embeddings.layer_norm.weight'])写差了
# 位置编码不是用输入词 id，而是位置序号 0,1,2,3...。token_type_embedding应该用全 0 的句子类型 id，不是词 id。
# 矩阵乘法用错（直接崩溃）torch.dot 只能算一维点积，多维必须用 torch.matmul 或 @。
# 除以 sqrt (d_k) 缺失（BERT 必须有）7. 实例化模型时没传输入，Bert 是模型类，不是函数！必须先实例化，再调用 forward：


# 位置编码没有 batch 广播 → 维度不匹配，加法错误
# 前馈网络的 LayerNorm 用错 → 应该作用在 transform_output + attn_output 上
# 池化取错 token → BERT 池化只取 第一个 token，不是整句

# 1. 词嵌入 + 修正位置编码（必须广播到batch维度）
#         we = self.word_embedding(x)
#         pos_ids = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
#         pos_ids = pos_ids.unsqueeze(0).expand(batch_size, seq_len)  # 关键修复1
#         pe = self.position_embedding(pos_ids)
#         se = self.token_embedding(torch.zeros_like(x))

# 3. 池化：BERT只取第一个token  关键修复5
#         pooled_output = self.tanh(self.pool_layer(attn_out[:, 0, :]))

# 前馈残差 + 层归一化  关键修复4
#         ff_out = self.ff_layer_norm(ff_out + attn_output)

# odict_keys(['embeddings.word_embeddings.weight', 'embeddings.position_embeddings.weight', 'embeddings.token_type_embeddings.weight',
#             'embeddings.LayerNorm.weight', 'embeddings.LayerNorm.bias', 'encoder.layer.0.attention.self.query.weight',
#             'encoder.layer.0.attention.self.query.bias', 'encoder.layer.0.attention.self.key.weight', 'encoder.layer.0.attention.self.key.bias',
#             'encoder.layer.0.attention.self.value.weight', 'encoder.layer.0.attention.self.value.bias', 'encoder.layer.0.attention.output.dense.weight',
#             'encoder.layer.0.attention.output.dense.bias', 'encoder.layer.0.attention.output.LayerNorm.weight',
#             'encoder.layer.0.attention.output.LayerNorm.bias', 'encoder.layer.0.intermediate.dense.weight', 'encoder.layer.0.intermediate.dense.bias',
#             'encoder.layer.0.output.dense.weight', 'encoder.layer.0.output.dense.bias', 'encoder.layer.0.output.LayerNorm.weight',
#             'encoder.layer.0.output.LayerNorm.bias', 'pooler.dense.weight', 'pooler.dense.bias'])

class Bert(nn.Module):
    def __init__(self, weights, hidden_size=768):
        super(Bert, self).__init__()
        self.head_num = 12

        self.word_embedding = nn.Embedding(21128, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(512, hidden_size, padding_idx=0)
        self.token_embedding = nn.Embedding(2, hidden_size, padding_idx=0)
        # 上面要加起来再过layer_norm
        self.embedding_layer_norm = nn.LayerNorm(768, )

        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        # 得到qk
        self.softmax = nn.Softmax(dim=-1)
        # 得到qkv
        self.attention = nn.Linear(hidden_size, hidden_size)
        # 残差后再layer_norm
        self.attention_norm = nn.LayerNorm(hidden_size)

        self.ff1 = nn.Linear(hidden_size, 4*hidden_size)
        self.gelu = nn.GELU()
        self.ff2 = nn.Linear(4*hidden_size, hidden_size)
        # 加残差再layer_norm
        self.ff_layer_norm = nn.LayerNorm(hidden_size)

        self.pool_layer = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()

        self.load_weight(weights)


    def forward(self, x, hidden_size=768, layer_num=1):
        we = self.word_embedding(x)
        pe = self.position_embedding(torch.tensor([i for i in range(x.size(1))]))

        se = self.token_embedding(torch.zeros_like(x))
        x = we+pe+se
        x = self.embedding_layer_norm(x)
        ff = self.all_transform(x, hidden_size)
        # 只取句首的特殊token
        pooled_output = self.pool_layer(ff[:, 0, :])
        pooled_output = self.tanh(pooled_output)
        return ff, pooled_output



    def multi_head(self, x):
        x = x.reshape(x.shape[0], x.shape[1], self.head_num, -1)
        x = x.transpose(1, 2)
        return x

    def all_transform(self, x, hidden_size):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = self.multi_head(q)
        k = self.multi_head(k)
        v = self.multi_head(v)

        qk = torch.matmul(q, k.transpose(2, 3))/ math.sqrt((hidden_size/self.head_num))
        qk = self.softmax(qk)
        qkv = qk @ v
        qkv = qkv.transpose(1, 2).reshape(x.shape[0], -1, hidden_size)

        attn_output = self.attention(qkv)
        attn_output = self.attention_norm((attn_output + x))

        ff = self.ff1(attn_output)
        ff = self.gelu(ff)
        transform_output = self.ff2(ff)

        # 这里一定要加attn_output，而不能是x
        transform_output = self.ff_layer_norm(transform_output + attn_output)

        return transform_output

# 都不用转置

    def load_weight(self, weights):
        self.word_embedding.weight.data.copy_(weights['embeddings.word_embeddings.weight'])
        self.position_embedding.weight.data.copy_(weights['embeddings.position_embeddings.weight'])
        self.token_embedding.weight.data.copy_(weights['embeddings.token_type_embeddings.weight'])
        # odict_keys(['embeddings.word_embeddings.weight', 'embeddings.position_embeddings.weight', 'embeddings.token_type_embeddings.weight',
        #             'embeddings.LayerNorm.weight', 'embeddings.LayerNorm.bias', 'encoder.layer.0.attention.self.query.weight',
        #             'encoder.layer.0.attention.self.query.bias', 'encoder.layer.0.attention.self.key.weight', 'encoder.layer.0.attention.self.key.bias',
        #             'encoder.layer.0.attention.self.value.weight', 'encoder.layer.0.attention.self.value.bias', 'encoder.layer.0.attention.output.dense.weight',
        #             'encoder.layer.0.attention.output.dense.bias', 'encoder.layer.0.attention.output.LayerNorm.weight',
        #             'encoder.layer.0.attention.output.LayerNorm.bias', 'encoder.layer.0.intermediate.dense.weight', 'encoder.layer.0.intermediate.dense.bias',
        #             'encoder.layer.0.output.dense.weight', 'encoder.layer.0.output.dense.bias', 'encoder.layer.0.output.LayerNorm.weight',
        #             'encoder.layer.0.output.LayerNorm.bias', 'pooler.dense.weight', 'pooler.dense.bias'])

        self.embedding_layer_norm.weight.data.copy_(weights['embeddings.LayerNorm.weight'])
        self.embedding_layer_norm.bias.data.copy_(weights['embeddings.LayerNorm.bias'])
        self.q.weight.data.copy_(weights['encoder.layer.0.attention.self.query.weight'])
        self.q.bias.data.copy_(weights['encoder.layer.0.attention.self.query.bias'])
        self.k.weight.data.copy_(weights['encoder.layer.0.attention.self.key.weight'])
        self.k.bias.data.copy_(weights['encoder.layer.0.attention.self.key.bias'])
        self.v.weight.data.copy_(weights['encoder.layer.0.attention.self.value.weight'])
        self.v.bias.data.copy_(weights['encoder.layer.0.attention.self.value.bias'])
        self.attention.weight.data.copy_(weights['encoder.layer.0.attention.output.dense.weight'])
        self.attention.bias.data.copy_(weights['encoder.layer.0.attention.output.dense.bias'])
        self.attention_norm.weight.data.copy_(weights['encoder.layer.0.attention.output.LayerNorm.weight'])
        self.attention_norm.bias.data.copy_(weights['encoder.layer.0.attention.output.LayerNorm.bias'])
        self.ff1.weight.data.copy_(weights['encoder.layer.0.intermediate.dense.weight'])
        self.ff1.bias.data.copy_(weights['encoder.layer.0.intermediate.dense.bias'])
        self.ff2.weight.data.copy_(weights['encoder.layer.0.output.dense.weight'])
        self.ff2.bias.data.copy_(weights['encoder.layer.0.output.dense.bias'])
        self.ff_layer_norm.weight.data.copy_(weights['encoder.layer.0.output.LayerNorm.weight'])
        self.ff_layer_norm.bias.data.copy_(weights['encoder.layer.0.output.LayerNorm.bias'])
        self.pool_layer.weight.data.copy_(weights['pooler.dense.weight'])
        self.pool_layer.bias.data.copy_(weights['pooler.dense.bias'])

bert.eval()
x = np.array([2450, 15486, 102, 2110])   #假想成4个字的句子
torch_x = torch.LongTensor([x])          #pytorch形式输入

seqence_output, pooler_output = bert(torch_x)
diy_bert = Bert(state_dict)
diy_bert.eval()
diy_seqence_output, diy_pooler_output = diy_bert(torch_x)

print("seqence_output",seqence_output,"\npooler_output", pooler_output,"\ndiy_seqence_output",diy_seqence_output, "\ndiy_pooler_output",diy_pooler_output)

