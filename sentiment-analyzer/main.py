from fastapi import FastAPI
from pydantic import BaseModel # Import BaseModel for request data validation
from transformers import pipeline # Import the pipeline function from transformers
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import os 

# --- 1. Create a FastAPI app instance ---
app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. Load the sentiment analysis model ---
# This line downloads and loads a pre-trained model.
# It's resource-intensive, so we do it once when the app starts.
print("Loading sentiment analysis model...")
sentiment_pipeline = pipeline("sentiment-analysis")
print("Model loaded successfully!")


# --- 3. Define the request data model ---
# This tells FastAPI what kind of data to expect in the request body.
# We expect a JSON object with a single key "text" which is a string.
class SentimentRequest(BaseModel):
    text: str


# --- MODIFIED ROOT ENDPOINT ---
# This endpoint will now serve your HTML page
@app.get("/", response_class=HTMLResponse)
async def read_index():
    # Construct the full path to the index.html file
    html_file_path = os.path.join("templates", "index.html")
    with open(html_file_path) as f:
        return HTMLResponse(content=f.read(), status_code=200)


# --- 5. Define the sentiment analysis endpoint ---
# We use @app.post() because the user is SENDING data to our API.
@app.post("/analyze-sentiment/")
def analyze_sentiment(request: SentimentRequest):
    
    result = sentiment_pipeline(request.text)[0]
    return result