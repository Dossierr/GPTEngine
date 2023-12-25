from fastapi import FastAPI, HTTPException, APIRouter
from chat_processing import process_query, index_files
import shutil
from rq import Queue
import redis
import time
import environ

env = environ.Env()
environ.Env.read_env()

router = APIRouter()


app = FastAPI()


# Set up Redis connection
#redis_conn = Redis(host='localhost', port=6379, db=0)
redis_conn = redis.Redis(
  host='redis-15281.c300.eu-central-1-1.ec2.cloud.redislabs.com',
  port=15281,
  password=env('REDIS_PASSWORD'),
  decode_responses=True,
  )

# Set up RQ queue
queue = Queue(connection=redis_conn)

def background_task():
    # Your background task logic here
    print("Task started...")
    time.sleep(10)  # Wait for 10 seconds
    print("Task completed!")

@app.get("/q/")
def read_root():
    return {"Hello": "World"}


@app.get("/q/task/")
def test_task():
    job = queue.enqueue(background_task)
    print(job)
    return {"message enqueued: ": job.id }

@app.post("/q/query")
async def post_query(request_data: dict):
    """ Example cURL call:
    curl --location 'http://localhost:8000/q/query' \
        --header 'Content-Type: application/json' \
        --data '{
            "dossier_id": "testfolder",
            "query": "Wie is de lijsttrekker van die partij?"
        }'
    """
    dossier_id = request_data.get("dossier_id")
    query = request_data.get("query")

    if not dossier_id or not query:
        raise HTTPException(status_code=400, detail="Both dossier_id and query are required.")

    return process_query(query, dossier_id)

    
@app.post("/q/reindex_dossier/{case_id}")
async def remove_folder(case_id: str):
    """
    curl -X POST http://localhost:8000/remove_folder/my_folder
    """
    try:
        shutil.rmtree("persist/"+case_id)
        index_files(case_id, True)
        return {"message": f"Folder {case_id} reindexed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))