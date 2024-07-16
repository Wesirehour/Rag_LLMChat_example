# Description

A simple example: Using Rag technology to converse with ChatLLM on a specific database
The embedded model is [BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5)
LLM can choose [ChatBaichuan](https://www.baichuan-ai.com/home), [ChatTongyi](https://help.aliyun.com/zh/dashscope/create-a-chat-foundation-model?spm=a2c4g.11186623.0.0.bd3c17d9Mbk2z6), [ChatSparkLLM](https://xinghuo.xfyun.cn/sparkapi)

# Install

1. install packages

```shell
pip install requirements.txt
```

# Start

1. construct database

```shell
python construct_database.py
```

2. run

```shell
python question.py --mode=0 # 1为测试模式，可以看到参考信息和运行速度
```

