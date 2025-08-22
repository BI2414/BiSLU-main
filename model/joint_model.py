import torch
import torch.nn as nn
from model.layer import *
from utils.utils import get_soft_slot
from transformers import AutoConfig
from model.layer.module import IntentSlotCoAttention

# 主模型，联合处理意图(ID)和槽位(SF)
class JointModel(nn.Module):
    def __init__(self, args, num_intent_labels, num_slot_labels):
        super(JointModel, self).__init__()
        self.args = args
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        self.num_intent_labels = num_intent_labels
        self.num_slot_labels = num_slot_labels
        self.wordrep = WordRep(args)

        self.soft_intent_classifier = IntentClassifier(
            config.hidden_size, self.num_intent_labels, args.dropout_rate)

        # --- 新增: Co-Attention Layer ---
        if args.use_co_attention:
            self.co_attention = IntentSlotCoAttention(
                hidden_size=config.hidden_size,  # BERT 的隐藏层大小
                intent_dim=num_intent_labels,  # 中间意图向量的维度
                proj_dim=args.co_attention_proj_dim  # 可以作为一个新参数传入，例如 150
            )
            # 使用协同注意力时，SlotClassifier的输入维度保持不变
            slot_input_dim = config.hidden_size
        else:
            # 不使用协同注意力时，保持原有逻辑
            slot_input_dim = config.hidden_size + (num_intent_labels if args.use_intent_context_attention else 0)

        self.slot_classifier = SlotClassifier(
            config,
            self.num_intent_labels,
            self.num_slot_labels,
            use_intent_context_attn=args.use_intent_context_attention and not args.use_co_attention,
            use_co_attention=args.use_co_attention,  # 新增参数
            max_seq_len=args.max_seq_length,
            dropout_rate=args.dropout_rate,
            hidden_dim_ffw=args.hidden_dim_ffw
        )
        if args.use_soft_slot:
            self.softmax = nn.Softmax(dim=-1)
            hard_intent_input_dim = config.hidden_size + num_slot_labels
            self.hard_intent_classifier = IntentClassifier(
                hard_intent_input_dim, self.num_intent_labels, args.dropout_rate)

    def forward(self, input_ids, attention_mask, words_lengths, word_attention_mask):

        cls_output, context_embedding = self.wordrep(
            input_ids, attention_mask, words_lengths)

        soft_intent_logits = self.soft_intent_classifier(cls_output)

        # --- 修改点: 应用协同注意力 ---
        if self.args.use_co_attention:
            # 使用协同注意力生成意图感知的词表示
            intent_aware_context = self.co_attention(context_embedding, soft_intent_logits)
            # 将处理后的词表示传入SlotClassifier，意图上下文设为None（因为已经融合）
            biaffine_score, segment_embedding = self.slot_classifier(
                intent_aware_context, None, word_attention_mask)
        else:
            # 保持原有逻辑
            biaffine_score, segment_embedding = self.slot_classifier(
                context_embedding, soft_intent_logits, word_attention_mask)

        if self.args.use_soft_slot:
            slot_label_feature = get_soft_slot(biaffine_score, word_attention_mask)
            slot_label_feature = self.softmax(slot_label_feature)

            intent_feature_concat = torch.cat(
                [cls_output, slot_label_feature], dim=-1)
            hard_intent_logits = self.hard_intent_classifier(
                intent_feature_concat)
            return cls_output, segment_embedding, soft_intent_logits, hard_intent_logits, biaffine_score
        else:
            return cls_output, segment_embedding, soft_intent_logits, biaffine_score
