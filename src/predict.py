from model import Bert_BiGru_Crf
from transformers import BertTokenizer
import config
import utlis


def stand_input(text, tokenizer):
    return tokenizer(text,
                     add_special_tokens=True,  # 添加special tokens， 也就是CLS和SEP
                     max_length=config.max_length,  # 设定最大文本长度
                     padding='max_length',  # padding
                     truncation=True,  # truncation
                     is_split_into_words=True,  # 是否分词
                     return_tensors='pt'  # 返回的类型为pytorch tensor
                     )


if __name__ == '__main__':
    text, label = utlis.read_data(config.test_path)
    text, label = text[0], label[0] # 取出测试集中第一条数据进行验证
    print(' '.join(text))
    print(label[1:len(text) + 1])
    # 加载预训练模型和tokenizer
    model = Bert_BiGru_Crf.from_pretrained(config.save_path)
    tokenizer = BertTokenizer.from_pretrained(config.save_path)
    # label和index互相转换
    label2index = {"P-B": 0, "P-I": 1, "T-B": 2, "T-I": 3, "A1-B": 4, "A1-I": 5, "A2-B": 6,
                   "A2-I": 7, "A3-B": 8, "A3-I": 9, "A4-B": 10, "A4-I": 11, "O": 12}

    index2label = {v: k for k, v in label2index.items()}
    # 获取tensor
    text = stand_input(text, tokenizer)
    # 获取decoder结果
    out = model.predict(text)
    print(out[0][1:-1])
    print([index2label[item] for item in out[0][1:-1]])
