import environ
import constants
import os
import redis
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from langchain.document_loaders import S3DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.memory import RedisChatMessageHistory
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.cache import RedisCache
from langchain.globals import set_llm_cache
from langchain.vectorstores.redis import Redis
from langchain.prompts import PromptTemplate


import numpy as np
import openai


# Get env variables
env = environ.Env()
environ.Env.read_env()
os.environ["REDIS_URL"] = constants.REDIS_URL
os.environ["OPENAI_API_KEY"] = constants.OPENAI_API_KEY
os.environ["REDIS_PASSWORD"] = constants.REDIS_PASSWORD
os.environ["S3_AWS_ACCESS_KEY_ID"] = constants.S3_AWS_ACCESS_KEY_ID
os.environ["S3_AWS_SECRET_KEY"] = constants.S3_AWS_SECRET_KEY
openai.api_key = env("OPENAI_API_KEY")

#Creating a Redis Instance
r = redis.Redis(
  host='redis-15281.c300.eu-central-1-1.ec2.cloud.redislabs.com',
  port=15281,
  password=env('REDIS_PASSWORD'),
  decode_responses=True)

#Uses Redis as cache for frequent queries witha time to live
set_llm_cache(RedisCache(r, ttl=60*60))


VECTOR_DIMENSIONS = 1536

def create_index(vector_dimensions: int, dossier_id: str):
    try:
        # check to see if index exists
        r.ft(dossier_id).info()
        print("Reusing existing index...")
    except:
        # schema
        schema = (
            TagField("tag"),                       # Tag Field Name
            VectorField("vector",                  # Vector Field Name
                "FLAT", {                          # Vector Index Type: FLAT or HNSW
                    "TYPE": "FLOAT32",             # FLOAT32 or FLOAT64
                    "DIM": vector_dimensions,      # Number of Vector Dimensions
                    "DISTANCE_METRIC": "COSINE",   # Vector Search Distance Metric
                }
            ),
        )

        # index Definition
        definition = IndexDefinition(prefix=[dossier_id], index_type=IndexType.HASH)

        # create Index
        return r.ft(dossier_id).create_index(fields=schema, definition=definition)


def indexing_folder(dosier_id):
    #Loading S3 bucket
    loader = S3DirectoryLoader("dossierr", 
                               prefix=dosier_id, 
                               aws_access_key_id=env('S3_AWS_ACCESS_KEY_ID'),
                               aws_secret_access_key=env('S3_AWS_SECRET_KEY'),
                               )
    
    docs = loader.load()
    #Splitting eaach document in chunks for better retrieval later
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
        )
    texts = text_splitter.split_documents(docs)

    # List of all file paths
    file_paths = [str(text.metadata['source']) for text in texts]
    # List of all text contents
    texts = [str(text.page_content) for text in texts]
    
    response = openai.Embedding.create(input=texts, model="text-embedding-ada-002")
    embeddings = np.array([r["embedding"] for r in response["data"]], dtype=np.float32)

    # Write to Redis
    pipe = r.pipeline()
    for i, embedding in enumerate(embeddings):
        key = f"{dosier_id}:{i}"
        pipe.hset(key, mapping={
            "vector": embedding.tobytes(),
            "content": texts[i],
            "tag": dosier_id,
            "document": file_paths[i],
        })
        pipe.expire(key, 60*60)
    pipe.execute()
    return None


"""def retrieve_documents(dossier_id, query):  
    # create query embedding
    response = openai.Embedding.create(input=[query], engine="text-embedding-ada-002")
    query_embedding = np.array([r["embedding"] for r in response["data"]], dtype=np.float32)[0]
    

    # query for similar documents that have the corresponding dossier_id tag
    query = (
        Query("(@tag:{"+dossier_id+" })=>[KNN 2 @vector $vec as score]")
        .sort_by("score")
        .return_fields("content", "tag", "score", "document")
        .paging(0, 2)
        .dialect(2)
    )

    query_params = {"vec": query_embedding.tobytes()}
    result = r.ft(dossier_id).search(query, query_params).docs

    #returns a string with context
    response = []
    sources = []
    for document in result:
        content = getattr(document, "content", "No content available")
        response.append(content)
        source = getattr(document, "document", "No source available")
        source = source.split("/")[-1] #just the file name
        sources.append(source)
    return response, sources"""

"""def answer_query_with_llm(dossier_id, query):
    #Finding a place to store the history
    chat_history = RedisChatMessageHistory(
            url=env('REDIS_URL'),
            session_id='-history',
            key_prefix=dossier_id,
            ttl=3600*24)
    documents, sources = retrieve_documents(dossier_id, query)
    docs = []

    for i, doc in enumerate(documents):  # Use enumerate to get both index and document
        document = Document(
            page_content=doc,
            metadata={'sources': sources[i]}  # Use the index to get the corresponding source
        )
        docs.append(document)
    """
"""doc_prompt = PromptTemplate.from_template("{page_content}")
    chain = (
        {
            "content": lambda docs: "\n\n".join(
                format_document(doc, doc_prompt) for doc in docs
            )
        }
        | PromptTemplate.from_template("Summarize the following content:\n\n{content}")
        | OpenAI(temperature=0.9)
        | StrOutputParser()
    )

    result = chain.invoke(docs)
    chat_history.add_user_message(query)
    chat_history.add_ai_message(result)
    print(result)"""
    
"""    ### NEW SECTION
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain

    # Create a prompt template
    prompt = (
        PromptTemplate.from_template("Tell me a joke about {topic}")
        + ", make it funny"
        + "\n\nand in {language}"
    )

    # Use the format method to set values
    formatted_prompt = prompt.format({topic='sports', language='spanish'})

    # Create a language model
    model = ChatOpenAI()

    # Create an LLMChain with the formatted prompt
    chain = LLMChain(llm=model, prompt=formatted_prompt)

    # Run the chain with arguments
    result = chain.run()

    # Print the result
    print(result)




    
    return None


DOSSIER_ID = 'testfolder'
#create_index(vector_dimensions=VECTOR_DIMENSIONS, dossier_id=DOSSIER_ID)
#indexing_folder(DOSSIER_ID)
query = 'Wat zegt de VVD over klimaat? '
result = answer_query_with_llm(DOSSIER_ID, query)


"""




