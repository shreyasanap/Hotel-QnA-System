import faiss
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, Query
from pydantic import BaseModel
import uvicorn
import time

df = pd.read_csv("cleaned_booking_data.csv")

with open("embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

index = faiss.read_index("faiss_index.bin")

app = FastAPI()

# Request model
class QueryRequest(BaseModel):
    query: str

def search_faiss(query_embedding, top_k=5):
    distances, indices = index.search(np.array([query_embedding]), top_k)
    return indices[0]

@app.post("/analytics")
def get_analytics(metric: str = Query(..., description="Metric to analyze")):
    """Returns analytics based on the requested metric."""
    if metric == "total_revenue_july_2017":
        total_revenue = df[(df['arrival_date_year'] == 2017) & (df['arrival_date_month'] == "July")]['adr'].sum()
        return {"total_revenue": total_revenue}

    if metric == "highest_cancellations":
        cancellations = df[df["is_canceled"] == 1]["hotel"].value_counts().to_dict()
        return {"highest_cancellations": cancellations}

    if metric == "average_booking_price":
        avg_price = df["adr"].mean()
        return {"average_price": avg_price}

    return {"error": "Invalid metric"}

@app.post("/ask")

def ask_question(request: QueryRequest):
    """Answers booking-related questions using FAISS."""
    start_time = time.time()

    try:
        # Simulated embedding -  matches FAISS dimensions
        query_embedding = np.random.rand(index.d)  
        
        #  Check embedding shape
        print(f"Query Embedding Shape: {query_embedding.shape}")

        # FAISS search
        retrieved_indices = search_faiss(query_embedding)

        #  Check retrieved indices
        print(f"Retrieved Indices: {retrieved_indices}")

        # Retrieve records safely
        if len(retrieved_indices) == 0:
            return {"error": "No matching results found."}

        results = df.iloc[retrieved_indices].to_dict(orient="records")

        response_time = time.time() - start_time
        return {
            "retrieved_results": results,
            "response_time": f"{response_time:.4f} seconds"
        }

    except Exception as e:
        print(f"Error in /ask: {e}")
        return {"error": "Internal Server Error", "details": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
