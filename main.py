from fastapi import FastAPI, HTTPException
from chat_processing import process_query
import shutil
from rq import Queue
from redis import Redis
import time

app = FastAPI(root_path="/q")

# Set up Redis connection
redis_conn = Redis(host='localhost', port=6379, db=0)

# Set up RQ queue
queue = Queue(connection=redis_conn)

def background_task():
    # Your background task logic here
    print("Task started...")
    time.sleep(10)  # Wait for 10 seconds
    print("Task completed!")

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/task/")
def test_task():
    job = queue.enqueue(background_task)
    print(job)
    return {"message enqueued: ": job.id }

@app.post("/query")
async def post_query(request_data: dict):
    """ Example cURL call:
    curl --location 'http://localhost:8000/query' \
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


"""@app.post("/query2")
async def post_query(request_data: dict):
    dossier_id = request_data.get("dossier_id")
    query = request_data.get("query")

    if not dossier_id or not query:
        raise HTTPException(status_code=400, detail="Both dossier_id and query are required.")

    return answer_query_with_llm(dossier_id,query)"""

    
@app.post("/reindex_dossier/{folder_path}")
async def remove_folder(folder_path: str):
    """
    curl -X POST http://localhost:8000/remove_folder/my_folder
    """
    try:
        shutil.rmtree("persist/"+folder_path)
        return {"message": f"Folder {folder_path} removed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))