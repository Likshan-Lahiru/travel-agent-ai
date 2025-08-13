import os
import json
import re
from typing import Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI

# Load env vars
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your_groq_api_key")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE", "https://api.groq.com/openai/v1")

# Create
# LLM instance
llm = ChatOpenAI(
    temperature=0.7,
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    max_tokens=1024
)


# Base system prompt
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


# Regex JSON extractor
def _extract_json(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text


# LLM call wrapper
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
            "llm_response": raw,  # Return raw so you can debug
            "prompt_list": [user_message],
            "travel_data_objects": {
                "experiences_object": [],
                "day_wise_itinerary_object": {}
            }
        }


# FastAPI app
app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# REST endpoint
@app.post("/chat")
async def chat_endpoint(payload: Dict[str, Any]):
    user_message = payload.get("message", "")
    return call_llm(user_message)


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            user_message = await websocket.receive_text()
            response = call_llm(user_message)
            await websocket.send_text(json.dumps(response))
    except WebSocketDisconnect:
        pass
