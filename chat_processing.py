import os
from langchain.globals import set_llm_cache
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.document_loaders import S3DirectoryLoader
from langchain.cache import InMemoryCache
from langchain.cache import RedisCache
from langchain.globals import set_llm_cache
import redis
import environ
from langchain.vectorstores import Chroma
from langchain.memory import RedisChatMessageHistory

env = environ.Env()
environ.Env.read_env()

# Create an instance of InMemoryCache
llm_cache = InMemoryCache()
set_llm_cache(llm_cache)

r = redis.Redis(
  host='redis-15281.c300.eu-central-1-1.ec2.cloud.redislabs.com',
  port=15281,
  password=env('REDIS_PASSWORD'),
  decode_responses=True)

#Uses Redis as cache for frequent queries witha time to live
set_llm_cache(RedisCache(r, ttl=60*60))

  
def process_query(query, dossier_id):  
    PERSIST = True #Persists data 
    chat_history = RedisChatMessageHistory(
        url=env('REDIS_URL'),
        session_id='-history',
        key_prefix=dossier_id,
        ttl=3600*24)
    

    if PERSIST and os.path.exists("persist/"+dossier_id):
        # So we have a folder with index and want to use it
        print("Reusing index:  "+dossier_id+"\n")
        vectorstore = Chroma(persist_directory="persist/"+dossier_id, embedding_function=OpenAIEmbeddings())
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        # No folder exists or we don't want to use it.
        s3_folder_path = str(dossier_id)
        loader = S3DirectoryLoader("dossierr", prefix=s3_folder_path, aws_access_key_id=env('S3_AWS_ACCESS_KEY_ID'),
                               aws_secret_access_key=env('S3_AWS_SECRET_KEY'))
        if PERSIST:
            #We reuse a index if it alreaady exists
            index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist/"+dossier_id}).from_loaders([loader])
        else:
            #We create a new index in a local folder
            index = VectorstoreIndexCreator().from_loaders([loader])
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(
        model="gpt-3.5-turbo",
        cache=True, temperature=1.3), 
        return_source_documents=True,
        # See documentation on retrievers: https://python.langchain.com/docs/modules/data_connection/retrievers/vectorstore 
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": 3}),
        )

    result = chain({"question": query, "chat_history": chat_history.messages})
    source_list = []
    for source in result["source_documents"]:
        source_name = source.metadata['source'].split("/")[-1]
        if source_name not in source_list:
            source_list.append(source_name)
    chat_history.add_user_message(query)
    chat_history.add_ai_message(result['answer'])
    return {'answer': result['answer'], 'sources':source_list}