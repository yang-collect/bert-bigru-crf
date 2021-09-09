from model import Bert_BiGru_Crf
from utlis import DataLoad
import torch
import numpy as np
from transformers import BertTokenizer
import config


def evaluate(path, tokenizer, model):
    val_data = DataLoad(tokenizer, path)
    avg_acc=[]
    with torch.no_grad():
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        for batch, data in enumerate(val_data):
            batch_content = {k: v.to(device) for k, v in data.items() if k != 'labels'}
            predictions = model.predict(batch_content)
            # 手动padding
            predictions = [item + [12] * (config.max_length - len(item)) if len(item) < config.max_length else item for
                           item in predictions]
            predictions=torch.tensor(predictions) # 转化为tensor
            acc=(predictions==data['labels']).sum()/predictions.numel() # 计算acc值 ，predictions.numel() shape的累积
            avg_acc.append(acc.numpy())
    return np.mean(avg_acc)


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained(config.save_path)
    model = Bert_BiGru_Crf.from_pretrained(config.save_path)

    print('train data  score:', evaluate(config.train_path, tokenizer, model))

    print('test data  score:', evaluate(config.test_path, tokenizer, model))

    print('dev data  score:', evaluate(config.dev_path, tokenizer, model))
