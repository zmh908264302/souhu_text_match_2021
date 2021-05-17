import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForPreTraining
from transformers.modeling_longformer import LongformerPreTrainedModel
from transformers.modeling_outputs import Seq2SeqSequenceClassifierOutput


class LongformerForClassification(LongformerPreTrainedModel):
    """
    数据元对标多任务模型，包括对象类词、表示词分类模型和特性词问答抽取模型
    """

    def __init__(self, config, model_dir):
        super(LongformerForClassification, self).__init__(config)
        self.longformer = AutoModelForPreTraining.from_pretrained(model_dir, config=config)
        self.classify = torch.nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            global_attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        # []
        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            # global_attention_mask=global_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 当前以seq_relationship_logits为分类结果，后面可以考虑最后一层隐藏层，可以看源码确认两者的区别
        logits = outputs.seq_relationship_logits

        # logits = self.classify(sequence_output)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        return Seq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )

    def predict(self, output):
        softmax = nn.Softmax(dim=1)
        y_pred_prob, y_pred = torch.max(softmax(output.data), 1)
        y_pred = y_pred.numpy()
        y_pred_prob = y_pred_prob.numpy()

        return y_pred_prob, y_pred
