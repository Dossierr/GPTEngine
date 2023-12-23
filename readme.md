# GPT Engine
The GPT engine processes user queries and returns text back to the user. It works on the principle of RAG (Retrieval Augmented Generation). The summary of this process is as follows: 

1. The user asks a query (what are ingredients for pizza)
2. That query is converted to a Vector 
3. That vector is compared to other vectors in our vector database (we use chroma and keep a database for each user in the Persist folder)
4. The GPT engine retrieves the top K relevant texts in the user documents. 
5. The GPT engine retrieves the last chat history 
6. The Chat history + relevant texts are sent to OpenAI 
7 A response is returned to the user. 