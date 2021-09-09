# 数据介绍

数据来源于百度飞桨的`paddlenlp`内置数据集，数据间的分隔符为`\002`，链接为：https://paddlenlp.bj.bcebos.com/paddlenlp/datasets/waybill.tar.gz

# 模型

`ner`任务属于`token classifer`任务，目前比较好的解决方案是采取`词向量+bilstm-crf` ，预训练的词向量可以引入很多先验信息，也在一定程度上缓解`oov词`的问题，双向`lstm`层用于学习输入数据双向的编码表示，`crf`则用于解决：`lstm的当前时刻输出没有考虑上一时刻的输出的问题`。词向量可以选择`word2vec`这类的静态词向量，也可以选择基于`bert`进行微调的动态词向量。

本文是基于[hugging face transformers](https://huggingface.co/transformers/training.html)实现的`bert-bigru-crf` ,参考https://github.com/HandsomeCao/Bert-BiLSTM-CRF-pytorch的实现，这里的`bert`采用的是[`bert-base-chinese`](https://huggingface.co/bert-base-chinese)是基于字的模型。

将其中的`pytorch_pretrained_bert`和`crf`部分替换为`transfomers`和[`pytorch-crf`](zhttps://pytorch-crf.readthedocs.io/en/stable/)

模型在经过15个epoch之后在测试集上的loss为：7.310302734375

这里取出训练集的第一条进行验证，

`/src/predict.py 输出如下：`

![image-20210909130203363](C:\Users\wie\AppData\Roaming\Typora\typora-user-images\image-20210909130203363.png)



`/src/server.py 在postman上测试结果如下：`

![image-20210909130428081](C:\Users\wie\AppData\Roaming\Typora\typora-user-images\image-20210909130428081.png)