import os
import json
import re
import random
from pathlib import Path
from typing import Dict, Any, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import json
from fastapi import WebSocket

# assumes pick_random_itinerary() is already defined as in the previous snippet
# Load env vars
load_dotenv()

# LLM (unchanged for /chat)
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0.7,
    max_tokens=1024,
    api_key=os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")  # fallback if needed
)

SYSTEM_PROMPT = """
You are TravelAgentAI, a helpful travel planner.
You MUST respond ONLY with a valid JSON object matching this schema:
{
    "llm_response": "string - a natural language summary",
    "prompt_list": ["array of related prompts or follow-up questions"],
    "travel_data_objects": {
        "experiences_object": [
            {
                "id": "string",
                "name": "string",
                "regions": [],
                "nationalExperiences": []
            }
        ],
        "day_wise_itinerary_object": {
            "day_id": {
                "day_id": "string",
                "timeslots": []
            }
        }
    }
}
Do not add any text before or after the JSON.
"""

def _extract_json(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text

def call_llm(user_message: str) -> Dict[str, Any]:
    strict_prompt = f"User message:\n{user_message}"
    completion = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": strict_prompt}
    ])
    raw = completion.content.strip()
    candidate = _extract_json(raw)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return {
            "llm_response": raw,
            "prompt_list": [user_message],
            "travel_data_objects": {
                "experiences_object": [],
                "day_wise_itinerary_object": {}
            }
        }

# -------- Random itinerary loader (for WebSocket) --------
FINAL_FILE_PATH = Path(os.getcwd()) / "final_itinerary.json"
_itinerary_cache: List[Any] = []

def load_itineraries(force_reload: bool = False) -> List[Any]:
    global _itinerary_cache
    if not _itinerary_cache or force_reload:
        if not FINAL_FILE_PATH.exists():
            raise FileNotFoundError(f"Missing file: {FINAL_FILE_PATH}")
        with FINAL_FILE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError("final_itinerary.json must be a non-empty JSON array.")
        _itinerary_cache = data
    return _itinerary_cache

def pick_random_itinerary() -> Any:
    items = load_itineraries()
    return random.choice(items)

# -------- FastAPI app --------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat_endpoint(payload: Dict[str, Any]):
    user_message = payload.get("message", "")
    return call_llm(user_message)

# Optional: simple REST endpoint to fetch a random itinerary (useful for testing)
@app.get("/itinerary/random")
async def get_random_itinerary():
    try:
        return pick_random_itinerary()
    except Exception as e:
        return {"error": str(e)}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        item = pick_random_itinerary()
        await websocket.send_text(json.dumps(item))  # send JSON as text
    except Exception as e:
        await websocket.send_text(json.dumps({"error": str(e)}))
    finally:
        await websocket.close()


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=True)
