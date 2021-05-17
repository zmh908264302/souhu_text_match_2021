import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel

from config import args


class TextClassification(BertPreTrainedModel):
    """
    数据元对标多任务模型，包括对象类词、表示词分类模型和特性词问答抽取模型
    """

    def __init__(self, config, fit_size=768):
        super(TextClassification, self).__init__(config)
        self.object_num_labels = len(args.labels_obj)
        self.express_num_labels = len(args.labels_express)

        self.bert = BertModel(config)

        self.obj_dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.obj_conv = nn.Linear(config.hidden_size, args.obj_hidden_size)
        self.obj_classify = nn.Linear(config.hidden_size, self.object_num_labels)

        self.fit_dense = nn.Linear(config.hidden_size, fit_size)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, is_student=False):
        # [batch_size, sequence_length, hidden_size], [batch_size, hidden_size]
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=True,
            output_hidden_states=True

        )
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        hidden_states = outputs[2]
        atts = outputs[3]

        # judge weather container text_b in text_a use first encoder input, that's [CLS]
        obj_classify = self.obj_classify(self.obj_dropout(pooled_output))

        tmp = []
        if is_student:
            for s_id, sequence_layer in enumerate(hidden_states):
                tmp.append(self.fit_dense(sequence_layer))
            hidden_states = tmp

        return obj_classify, atts, hidden_states

    def predict(self, output):
        softmax = nn.Softmax(dim=1)
        y_pred_prob, y_pred = torch.max(softmax(output.data), 1)
        y_pred = y_pred.numpy()
        y_pred_prob = y_pred_prob.numpy()

        return y_pred_prob, y_pred


class DataElementClassification(BertPreTrainedModel):
    """
    数据元对标多任务模型，包括对象类词、表示词分类模型和特性词问答抽取模型
    """

    def __init__(self, config):
        super(DataElementClassification, self).__init__(config)
        self.num_de = 2

        self.bert = BertModel(config)

        self.obj_dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.obj_conv = nn.Linear(config.hidden_size, args.obj_hidden_size)
        self.classify = nn.Linear(config.hidden_size, self.num_de)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, is_student=False):
        # [batch_size, sequence_length, hidden_size], [batch_size, hidden_size]
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=True,
            output_hidden_states=True

        )
        # sequence_output = outputs[0]
        pooled_output = outputs[1]
        hidden_states = outputs[2]
        atts = outputs[3]

        # judge weather container text_b in text_a use first encoder input, that's [CLS]
        classify = self.classify(self.obj_dropout(pooled_output))

        return classify, atts, hidden_states

    def predict(self, output):
        softmax = nn.Softmax(dim=1)
        y_pred_prob, y_pred = torch.max(softmax(output.data), 1)
        y_pred = y_pred.numpy()
        y_pred_prob = y_pred_prob.numpy()

        return y_pred_prob, y_pred
