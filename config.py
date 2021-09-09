from pathlib import Path

project_path = Path(__file__).parent

train_path = project_path.joinpath('./data/train.txt')
test_path = project_path.joinpath('./data/test.txt')
dev_path = project_path.joinpath('./data/dev.txt')
word_dic_path = project_path.joinpath('./data/word.dic')
tag_dic_path = project_path.joinpath('./data/tag.dic')

batch_size = 32

model_path = r'C:\Users\wie\Documents\pretrain_model\bert-base-chinese'

max_length = 67  # 训练和测试集最大字符长度均为65，加上开头和结尾的token

tag_map = project_path.joinpath('./data/tag_map.json')

label_num = 13

save_path = r'C:\Users\wie\Documents\项目\model_file\bert-bigru-crf'

epochs = 15

num_warmup_steps = 200

need_rnn = True

metric_path = str(project_path.joinpath('f1.py'))
