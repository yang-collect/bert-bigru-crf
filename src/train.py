import numpy as np
import torch
from transformers import AdamW, get_scheduler, BertTokenizer
import argparse
import os
import sys

abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abs_path)
import model
import utlis, config


def parse():
    # 用于控制部分参数解析，便于在终端中直接传入指定值进行修改
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--save_path', type=str, default=str(config.save_path))
    arg('--model_path', type=str, default=str(config.model_path),
        help='pretrained model path,we need to use it fine-tune our data')
    arg('--train_path', type=str, default=str(config.train_path))
    arg('--test_path', type=str, default=str(config.test_path))
    arg('--epochs', type=int, default=config.epochs)

    args = parser.parse_args()
    return args


def compute_loss(model, val_data):
    """Evaluate the loss and f1 score for an epoch.

    Args:
        model (torch.nn.Module): The model to evaluate.
        val_data (dataset.PairDataset): The evaluation data set.

    Returns:
        numpy ndarray: The average loss of the dev set.
    """
    print('validating')
    # metric = load_metric("f1")
    # metric = load_metric("f1")
    val_loss = []
    with torch.no_grad():
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        for batch, data in enumerate(val_data):
            batch_content = {k: v.to(device) for k, v in data.items() if k != 'labels'}
            loss = model(batch_content, data['labels'].to(device))
        val_loss.append(loss.item())

    return np.mean(val_loss)


def train(args):
    # 加载预训练模型和tokenizer
    embed_model = model.Bert_BiGru_Crf.from_pretrained(args.model_path)
    # model = BertModel.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(args.model_path)

    # 加载训练和测试集
    train_dataloder = utlis.DataLoad(tokenizer, args.train_path)
    test_dataloder = utlis.DataLoad(tokenizer, args.test_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 实例化优化器，
    optimizer = AdamW(embed_model.parameters(), lr=1e-4)

    num_training_steps = args.epochs * len(train_dataloder)
    # warm up
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # 模型训练
    val_loss = np.inf
    embed_model.to(device)
    embed_model.train()

    for epoch in range(args.epochs):
        batch_loss = []
        for num, batch in enumerate(train_dataloder):
            # 分别加载 query 、title、 label
            batch_content = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            loss = embed_model(batch_content, batch['labels'].to(device))
            # batch_label = batch['labels'].to(device)
            # loss = loss_func(prob, batch_label)
            batch_loss.append(loss.item())
            # 梯度更新
            loss.backward()
            # 优化器和学习率更新
            optimizer.step()
            lr_scheduler.step()
            # 梯度清零
            optimizer.zero_grad()
            # 每15个打印一次结果
            if num % 15 == 0:
                print(f'epoch:{epoch},batch :{num} ,train_loss :{loss} !')
        #
        epoch_loss = np.mean(batch_loss)
        avg_val_loss = compute_loss(embed_model, test_dataloder)
        print(f'epoch:{epoch},tran_loss:{epoch_loss},valid loss;{avg_val_loss}')
        print('*' * 100)
        # Update minimum evaluating loss.
        if avg_val_loss < val_loss:
            tokenizer.save_pretrained(args.save_path)
            embed_model.save_pretrained(args.save_path)
            val_loss = avg_val_loss

    print(val_loss)


if __name__ == '__main__':
    args = parse()
    train(args)
