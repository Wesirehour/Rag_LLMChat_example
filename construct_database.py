from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

# 加载文档
loader = TextLoader("./trip.txt", encoding="utf-8")
documents = loader.load()
print('加载完成')

# 文档分块
text_splitter = CharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=0,
    separator="\n"
)
documents = text_splitter.split_documents(documents)
print('分块完成')

# 初始化嵌入模型
# model_name = "moka-ai/m3e-base"
model_name = "BAAI/bge-large-zh-v1.5" # BGE嵌入
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embedding = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction="为这个句子生成表示以用于检索相关文章："
)
embedding.query_instruction = "为这个句子生成表示以用于检索相关文章："
print('初始化嵌入模型完成')

# Create and persist the vector database
persist_directory = 'trip_vector_database'
db = Chroma.from_documents(documents, embedding, persist_directory=persist_directory)
db.persist()
print(f"数据库中文档数量: {db._collection.count()}")
print('构建数据库完成')
