from model import Bert_BiGru_Crf
from utlis import DataLoad
import torch
import numpy as np
from transformers import BertTokenizer
import config


def evaluate(path, tokenizer, model):
    val_data = DataLoad(tokenizer, path)
    avg_recall=[]
    with torch.no_grad():
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        for batch, data in enumerate(val_data):
            batch_content = {k: v.to(device) for k, v in data.items() if k != 'labels'}
            predictions = model.predict(batch_content)
            # 手动padding
            predictions = [item + [12] * (config.max_length - len(item)) if len(item) < config.max_length else item for
                           item in predictions]
            predictions=torch.tensor(predictions) # 转化为tensor
            recall=((predictions==data['labels'])&(data['labels']!=12)).sum()/(data['labels']!=12).sum() # 计算acc值 ，predictions.numel() shape的累积
            avg_recall.append(recall.numpy())
    return np.mean(avg_recall)


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained(config.save_path)
    model = Bert_BiGru_Crf.from_pretrained(config.save_path)
    # 有效部分是指非O的部分
    print('train data 的有效部分召回为:', evaluate(config.train_path, tokenizer, model))

    print('test data 的有效部分召回为:', evaluate(config.test_path, tokenizer, model))

    print('dev data 的有效部分召回为:', evaluate(config.dev_path, tokenizer, model))
