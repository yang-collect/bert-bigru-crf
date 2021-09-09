from flask import Flask, request, Response, jsonify
from concurrent.futures import ThreadPoolExecutor
from transformers import BertTokenizer
import json
import datetime

import os
import sys

abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abs_path)

from src.model import Bert_BiGru_Crf
import config


class MyResponse(Response):
    @classmethod
    def force_type(cls, response, environ=None):
        if isinstance(response, (list, dict)):
            response = jsonify(response)
        return super(Response, cls).force_type(response, environ)


# 创建服务
server = Flask(__name__)
# 约定flask接受参数的类型
server.response_class = MyResponse
# 创建一个线程池，默认线程数量为cpu核数的5倍
executor = ThreadPoolExecutor()
# fine-tune模型路径
model_path = config.save_path
# 加载模型
model = Bert_BiGru_Crf.from_pretrained(model_path)
# tokenizer实例化
tokenizer = BertTokenizer.from_pretrained(model_path)

label2index = {"P-B": 0, "P-I": 1, "T-B": 2, "T-I": 3, "A1-B": 4, "A1-I": 5, "A2-B": 6,
               "A2-I": 7, "A3-B": 8, "A3-I": 9, "A4-B": 10, "A4-I": 11, "O": 12}

index2label = {v: k for k, v in label2index.items()}


def stand_input(text):
    return tokenizer(text.split(),
                     # add_special_tokens=True,  # 添加special tokens， 也就是CLS和SEP
                     max_length=config.max_length,  # 设定最大文本长度
                     padding='max_length',  # padding
                     truncation=True,  # truncation
                     is_split_into_words=True,  # 是否分词
                     return_tensors='pt'  # 返回的类型为pytorch tensor
                     )


# 绑定目录以及方法
@server.route('/ner/app', methods=["POST"])
def scene_object_appearance_class():
    data = request.get_json()
    # print(data['text'].split())
    output_res = {}
    if len(data) == 0:
        output_res["status"] = "400"
        output_res["msg"] = "Flase"
        output_res['label'] = "Flase"
        return output_res
    else:
        try:
            # 对传入的一条数据进行tokenizer
            text = stand_input(data['text'])
            out = model.predict(text)
            # 获取label对应的文本
            output_res['label'] = ' '.join([index2label[item] for item in out[0][1:-1]])
            return json.dumps(output_res, ensure_ascii=False)
        except Exception as e:
            print("异常原因: ", e)
            return {"error": 500}


def host():
    """ main 函数
    """
    HOST = '0.0.0.0'
    # 服务端口，为外部访问
    PORT = 5018
    server.config["JSON_AS_ASCII"] = False
    server.run(host=HOST, port=PORT, threaded=True)


if __name__ == "__main__":
    nowTime1 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 现在
    print("nowTime1: ", nowTime1)

    host()

    nowTime2 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 现在
    print("nowTime1: ", nowTime2)
