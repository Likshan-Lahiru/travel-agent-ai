import os
import json
import re
from typing import Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
# Create LLM instance (pick a Groq model you have access to)
from langchain_groq import ChatGroq
import os

# Load env vars
load_dotenv()

# Do NOT override OpenAI vars; use GROQ_API_KEY instead
# .env should contain: GROQ_API_KEY=xxxxx



llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0.7,
    max_tokens=1024,
    api_key=os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")  # <- fallback
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

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     try:
#         while True:
#             user_message = await websocket.receive_text()
#             response = call_llm(user_message)
#             await websocket.send_text(json.dumps(response))
#     except WebSocketDisconnect:
#         pass




if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=True)
