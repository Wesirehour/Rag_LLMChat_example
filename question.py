import time
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import argparse


def load_embedding_model():
    # 加载嵌入模型
    # model_name = "moka-ai/m3e-base"
    model_name = "BAAI/bge-large-zh-v1.5"  # BGE嵌入
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embedding = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        query_instruction="为这个句子生成表示以用于检索相关文章："
    )
    embedding.query_instruction = "为这个句子生成表示以用于检索相关文章："
    print('嵌入模型初始化完成')
    # time.sleep(10)
    persist_directory = 'trip_vector_database'
    db = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    print(f"数据库中文档数量: {db._collection.count()}")
    retriever = db.as_retriever(search_kwargs={'k': 1})  # 检索上下文的数量

    return retriever


def init_LLM():
    # 初始化prompt
    template = """你是文旅方向的资深导游。使用以下检索到的上下文来详细回答问题。如果你不知道答案，就说你不知道。Question: {question}    Context: {context}    
    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    # 初始化LLM
    from langchain_community.chat_models import ChatBaichuan, ChatTongyi, ChatSparkLLM
    llm = ChatBaichuan(baichuan_api_key='', temperature=0.3)
    # baichuan_api_key请查看https://platform.baichuan-ai.com/console/apikey

    # llm = ChatTongyi(model='qwen-max-longcontext', temperature=0.3)
    # Tongyi_qwen的key请查看https://help.aliyun.com/zh/dashscope/developer-reference/acquisition-and-configuration-of-api-key?

    # llm = ChatSparkLLM( spark_app_id='', spark_api_key='', spark_api_secret='' )
    # SparkLLM的key请查看https://xinghuo.xfyun.cn/sparkapi

    return prompt, llm


def question(retriever, prompt, llm):
    rag_chain = ({"context": retriever, "question": RunnablePassthrough()}
                 | prompt
                 | llm
                 | StrOutputParser())
    return rag_chain


# 计算耗费时间
def measure_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process mode options.')
    parser.add_argument('--mode', type=int, required=True, help='An integer mode value')
    args = parser.parse_args()
    mode = args.mode  # 输出模式
    retriever = load_embedding_model()
    prompt, llm = init_LLM()
    rag_chain = question(retriever, prompt, llm)
    while True:
        query = input("请输入问题 (输入'退出'以结束): ")
        if query.lower() == '退出':
            break
        if mode == 0:
            response = rag_chain.invoke(query)
            print(f"答案: {response}")
        else:
            # 测试速度
            # Measure time for retrieval
            retriever_result, retriever_time = measure_time(retriever.invoke, query)

            # Measure time for prompt generation
            prompt_input = {"context": retriever_result, "question": query}
            prompt_result, prompt_time = measure_time(prompt.format, **prompt_input)
            print('参考信息：')
            print(prompt_input)
            print('===============================')

            # Measure time for LLM response
            llm_result, llm_time = measure_time(llm.invoke, prompt_result)

            # Measure time for output parsing
            response, parser_time = measure_time(StrOutputParser().parse, llm_result)

            # Print the response and the times taken
            print(f"答案: {response}")
            print(f"Retrieval time: {retriever_time:.2f} seconds")
            print(f"Prompt generation time: {prompt_time:.2f} seconds")
            print(f"LLM response time: {llm_time:.2f} seconds")
            print(f"Output parsing time: {parser_time:.2f} seconds")
