import config
import json, torch
from torch.utils.data import Dataset, DataLoader

with open(config.tag_map, 'r', encoding='utf-8') as f:
    label_map = json.load(f)


def read_data(path=config.train_path):
    """ 读取数据文件，并返回text 列表和label列表

    :param path: 路径
    :return: text列表和label列表
    """
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        content, labels = [], []
        for i, line in enumerate(lines):
            if i > 0:
                text = line.split('\t')[0].split('\002')
                label = line.split()[1].split('\002')
                content.append(text)
                # 对label进行padding，将长度不足的进行补齐，并未做裁剪，padding的时候取O作为padding的填充值
                labels.append([label_map['O']] + [label_map[item] for item in label] + [label_map['O']] * (
                        config.max_length + 1 - len(label)))
    return content, labels


class MyDataset(Dataset):
    def __init__(self, data, tokenizer):
        """自定义datset类
        :param data:一个tuple，包含content和label
        :param tokenizer: tokenizer实例
        """
        self.tokenizer = tokenizer
        self.labels = torch.tensor(data[1])
        self.input_ids, self.attention_mask = self.encode(data[0])
        self.length = len(self.input_ids)

    def encode(self, text_list):
        token = self.tokenizer(text_list,
                               is_split_into_words=True,
                               padding='max_length',
                               truncation=True,
                               max_length=config.max_length + 2,
                               return_tensors='pt')

        return token['input_ids'], token['attention_mask']

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return {'input_ids': self.input_ids[index],
                'attention_mask': self.attention_mask[index],
                'labels': self.labels[index]}


def DataLoad(tokenizer, path=config.train_path):
    data = read_data(path)  # 读取数据
    dataset = MyDataset(data, tokenizer)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True)  # 返回datloder

# if __name__ == '__main__':
# #     from transformers import AutoModel, AutoTokenizer
# #
# #     model_path = config.model_path
# #     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     content, labels = read_data()
#     print(content[0])
#     print(labels[0])
# #     # max_len = max(map(len, labels))
# #     # print(max_len)
# #     dataloader = DataLoad(tokenizer)
# #     for num, batch in enumerate(dataloader):
# #         print(num,batch['input_ids'].size())
