from torchcrf import CRF
from torch import nn
from transformers import BertModel, BertPreTrainedModel
from config import need_rnn


class Bert_BiGru_Crf(BertPreTrainedModel):
    def __init__(self, config, need_birnn=need_rnn, rnn_dim=128, label_num=13):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.need_birnn = need_birnn
        if need_birnn:
            self.gru = nn.GRU(768,
                              rnn_dim,
                              num_layers=2,
                              bidirectional=True,
                              dropout=.3,
                              batch_first=True
                              )
            self.fc = nn.Linear(rnn_dim * 2, label_num)  # BOS EOS
        else:
            self.fc = nn.Linear(config.hidden_size, label_num)
        self.crf = CRF(label_num, batch_first=True)

    def forward(self, text, label):
        out = self.bert(input_ids=text['input_ids'], attention_mask=text['attention_mask']).last_hidden_state
        if self.need_birnn:
            out, _ = self.gru(out)
        out = self.dropout(out)
        output = self.fc(out)
        loss = -self.crf(output, label, mask=text['attention_mask'].byte())
        return loss

    def predict(self, text):
        """ 实现其predict接口

        :param text:  输入文本的input_ids、attention_mask
        :return:  decode解码的类别
        """
        out = self.bert(input_ids=text['input_ids'], attention_mask=text['attention_mask']).last_hidden_state
        if self.need_birnn:
            out, _ = self.gru(out)

        out= self.dropout(out)
        output = self.fc(out)
        pred = self.crf.decode(output, text['attention_mask'].byte())

        return pred
