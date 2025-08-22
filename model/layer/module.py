import numpy as np
import torch
import torch.nn as nn
from click.core import F

from model.layer import FeedforwardLayer, BiaffineLayer

# 意图分类器（线性层）
class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.0):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

# 槽位分类器（含Biaffine机制）
class SlotClassifier(nn.Module):
    def __init__(
            self,
            config,
            num_intent_labels,
            num_slot_labels,
            use_intent_context_attn=False,
            use_co_attention=False,
            use_span_self_attention=False,
            max_seq_len=50,
            dropout_rate=0.0,
            hidden_dim_ffw=300,
    ):
        super(SlotClassifier, self).__init__()
        self.use_intent_context_attn = use_intent_context_attn
        self.use_co_attention = use_co_attention  # 新增：协同注意力标志
        self.use_span_self_attention = use_span_self_attention  # NEW
        self.max_seq_len = max_seq_len
        self.num_intent_labels = num_intent_labels

        # 根据不同的意图融合方式确定输入维度
        if self.use_intent_context_attn:
            hidden_dim = config.hidden_size + num_intent_labels
            self.sigmoid = nn.Sigmoid()
        elif self.use_co_attention:
            hidden_dim = config.hidden_size
            # 新增：协同注意力层
            self.co_attention = IntentSlotCoAttention(
                hidden_size=config.hidden_size,
                intent_dim=num_intent_labels,
                proj_dim=hidden_dim_ffw // 2  # 使用FFN隐藏维度的一半作为投影维度
            )
        else:
            hidden_dim = config.hidden_size

        self.dropout = nn.Dropout(dropout_rate)

        self.feedStart = FeedforwardLayer(
            d_in=hidden_dim, d_hid=hidden_dim_ffw)
        self.feedEnd = FeedforwardLayer(d_in=hidden_dim, d_hid=hidden_dim_ffw)
        self.biaffine = BiaffineLayer(
            inSize1=hidden_dim, inSize2=hidden_dim, classSize=256)

        if self.use_span_self_attention:
            # The input dim to SA is 256 (output dim of the biaffine layer)
            self.span_self_attention = SpanSelfAttention(span_rep_dim=256, num_heads=4, dropout=dropout_rate)

        self.classifier = nn.Linear(256,num_slot_labels)

    def forward(self, word_context, intent_context,word_attention_mask=None):

        if self.use_intent_context_attn:
            intent_context = self.sigmoid(intent_context)
            intent_context = torch.unsqueeze(intent_context, 1)
            context = intent_context.repeat(1,word_context.shape[1],1)
            output = torch.cat((context, word_context), dim=2)
        elif self.use_co_attention:
            # 新增：使用协同注意力
            output = self.co_attention(word_context, intent_context)
        else:
            output = word_context
        x = self.dropout(output)
        start = self.feedStart(x)
        end = self.feedEnd(x)
        embedding = self.biaffine(start, end)
        embedding = self.dropout(embedding)

        if self.use_span_self_attention:
            batch_size, seq_len, _, feat_dim = embedding.shape
            flattened_embeddings = embedding.reshape(batch_size, -1, feat_dim)
            span_mask = None
            enhanced_embeddings = self.span_self_attention(flattened_embeddings, span_mask)
            embedding = enhanced_embeddings.reshape(batch_size, seq_len, seq_len, feat_dim)

        score = self.classifier(embedding)
        return score, embedding


class IntentSlotCoAttention(nn.Module):

    def __init__(self, hidden_size, intent_dim, proj_dim=None):
        super(IntentSlotCoAttention, self).__init__()
        if proj_dim is None:
            proj_dim = hidden_size // 2  # Default projection dimension

        # Project intent vector to a space that can be compared with word features
        self.intent_proj = nn.Linear(intent_dim, proj_dim)
        # Project word features to the same space
        self.word_proj = nn.Linear(hidden_size, proj_dim)
        # A learnable context vector to calculate attention scores
        self.context_vector = nn.Linear(proj_dim, 1, bias=False)
        # Layer to combine the original word feature and the intent-aware context
        self.combine_layer = nn.Linear(hidden_size + proj_dim, hidden_size)

        self.hidden_size = hidden_size
        self.proj_dim = proj_dim
        self.dropout = nn.Dropout(0.1)

    def forward(self, word_features, intent_vector):
        batch_size, seq_len, _ = word_features.shape

        projected_intent = self.intent_proj(intent_vector).unsqueeze(1)
        projected_intent = projected_intent.repeat(1, seq_len, 1)

        projected_words = self.word_proj(word_features)

        combined = torch.tanh(projected_words + projected_intent)

        attention_scores = self.context_vector(combined).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=-1)

        intent_context = projected_intent * attention_weights.unsqueeze(-1)

        combined_representation = torch.cat([word_features, intent_context],
                                            dim=-1)

        output = self.combine_layer(combined_representation)
        output = self.dropout(output)

        return output

class SpanSelfAttention(nn.Module):

    def __init__(self, span_rep_dim, num_heads=4, dropout=0.1):
        super(SpanSelfAttention, self).__init__()
        self.span_rep_dim = span_rep_dim
        self.num_heads = num_heads
        assert span_rep_dim % num_heads == 0, "span_rep_dim must be divisible by num_heads"
        self.head_dim = span_rep_dim // num_heads

        self.w_q = nn.Linear(span_rep_dim, span_rep_dim)
        self.w_k = nn.Linear(span_rep_dim, span_rep_dim)
        self.w_v = nn.Linear(span_rep_dim, span_rep_dim)

        self.attention_dropout = nn.Dropout(dropout)

        self.fc_out = nn.Linear(span_rep_dim, span_rep_dim)
        self.layer_norm1 = nn.LayerNorm(span_rep_dim)
        self.layer_norm2 = nn.LayerNorm(span_rep_dim)

        self.ffn = nn.Sequential(
            nn.Linear(span_rep_dim, span_rep_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(span_rep_dim * 2, span_rep_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, span_mask=None):

        batch_size, num_spans, _ = x.shape
        residual = x

        Q = self.w_q(x).view(batch_size, num_spans, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.w_k(x).view(batch_size, num_spans, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.w_v(x).view(batch_size, num_spans, self.num_heads, self.head_dim).transpose(1, 2)

        scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(x.device)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        if span_mask is not None:

            mask = span_mask.unsqueeze(1).unsqueeze(2)
            mask = mask.expand(-1, self.num_heads, num_spans, -1)
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        context = torch.matmul(attention_weights, V)

        context = context.transpose(1, 2).contiguous().view(batch_size, num_spans, self.span_rep_dim)
        output = self.fc_out(context)

        x = self.layer_norm1(residual + output)

        residual_ffn = x
        ffn_output = self.ffn(x)
        enhanced_x = self.layer_norm2(residual_ffn + ffn_output)

        return enhanced_x