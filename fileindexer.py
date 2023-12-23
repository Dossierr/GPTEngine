# General imports
import environ
import openai
import numpy as np
import os



# Redis
import redis
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from langchain.vectorstores.redis import Redis as RedisVectorStore

# Langchaing
from langchain.document_loaders import S3DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings



VECTOR_DIMENSIONS = 1536

env = environ.Env()
environ.Env.read_env()

openai.api_key = env('OPENAI_API_KEY')

r = redis.Redis(
    host='redis-15281.c300.eu-central-1-1.ec2.cloud.redislabs.com',
    port=15281,
    password=env('REDIS_PASSWORD'),
    decode_responses=True)

vectorstore = RedisVectorStore(env('REDIS_URL'),'testfolder', OpenAIEmbeddings)

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
        print(type(r.ft(dossier_id).create_index(fields=schema, definition=definition)))
        return None


def index_folder(dossier_id): 
    loader = S3DirectoryLoader("dossierr", 
                               prefix=dossier_id, 
                               aws_access_key_id=env('S3_AWS_ACCESS_KEY_ID'),
                               aws_secret_access_key=env('S3_AWS_SECRET_KEY'),
                               )
    
    docs = loader.load()
    #Splitting eaach document in chunks for better retrieval later
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=200
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
        key = f"{dossier_id}:{i}"
        pipe.hset(key, mapping={
            "vector": embedding.tobytes(),
            "content": texts[i],
            "tag": dossier_id,
            "document": file_paths[i],
        })
        pipe.expire(key, 60*60)
    pipe.execute()
    return None

def query_index(dossier_id):
    r = Redis(env('REDIS_URL'), dossier_id, OpenAIEmbeddings())
    results = rds.similarity_search("Milieu")
    print(results[0].page_content)
    




DOSSIER_ID = 'testfolder'
#create_index(vector_dimensions=VECTOR_DIMENSIONS, dossier_id=DOSSIER_ID)
#index_folder(DOSSIER_ID)
query_index(DOSSIER_ID)
