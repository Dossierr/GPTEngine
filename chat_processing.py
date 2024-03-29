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
from langchain.callbacks import get_openai_callback
import redis
import environ
from langchain.vectorstores import Chroma
from langchain.memory import RedisChatMessageHistory
from tasks.billing import bill_tokens
from rq import Queue


env = environ.Env()
environ.Env.read_env()




r = redis.Redis(
  host='redis-15281.c300.eu-central-1-1.ec2.cloud.redislabs.com',
  port=15281,
  password=env('REDIS_PASSWORD'),
  decode_responses=True)

queue = Queue(connection=r)



#Uses Redis as cache for frequent queries witha time to live
set_llm_cache(RedisCache(r, ttl=60*60))

def index_files(dossier_id, PERSIST):
    print("creating new index for: ", dossier_id)
    # No folder exists or we don't want to use it.
    s3_folder_path = 'cases/'+str(dossier_id)
    loader = S3DirectoryLoader("dossierr", prefix=s3_folder_path, aws_access_key_id=env('S3_AWS_ACCESS_KEY_ID'),
                        aws_secret_access_key=env('S3_AWS_SECRET_KEY'))
    if PERSIST:
        #We reuse a index if it alreaady exists
        index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist/"+dossier_id}).from_loaders([loader])
    else:
        #We create a new index in a local folder
        index = VectorstoreIndexCreator().from_loaders([loader])
    return index

  
def process_query(query, dossier_id, billing_token):  
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
        index = index_files(dossier_id, PERSIST)
        
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(
        model="gpt-3.5-turbo",
        cache=True, 
        temperature=1.3), 
        return_source_documents=True,
        # See documentation on retrievers: https://python.langchain.com/docs/modules/data_connection/retrievers/vectorstore 
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": 3}),
        )
    #WE take the last 3 messages so we don't exceed the context (In the response we send along all messages)
    short_history = chat_history.messages[-3:]
    with get_openai_callback() as cb:
        result = chain({"question": query, "chat_history": short_history})
    tokens_used = {'total':cb.total_tokens, 'prompt_and_context':cb.prompt_tokens, 'response':cb.completion_tokens}
    job = queue.enqueue(bill_tokens, billing_token, cb.total_tokens)
    source_list = []
    for source in result["source_documents"]:
        source_name = source.metadata['source'].split("/")[-1]
        if source_name not in source_list:
            source_list.append(source_name)
    chat_history.add_user_message(query)
    chat_history.add_ai_message(result['answer'])
    return {'chat': chat_history.messages, 'sources':source_list, 'tokens':tokens_used}