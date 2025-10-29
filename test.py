import asyncio
import json
import requests
import websockets

# Base URL of your deployed FastAPI app
BASE_URL = "https://travel-agent-ai.onrender.com"
WS_URL = f"wss://travel-agent-ai.onrender.com/ws"


# ---------- Test /chat HTTP endpoint ----------
def test_chat_endpoint():
    payload = {"message": "Plan a 7-day honeymoon trip to Paris."}
    print("🛰️ Sending HTTP POST to /chat ...")
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    if response.status_code == 200:
        data = response.json()
        print(" /chat Response:\n", json.dumps(data, indent=4))
    else:
        print(" /chat Request Failed:", response.status_code, response.text)


# ---------- Test /ws WebSocket endpoint ----------
async def test_ws_endpoint():
    print("\n🌐 Connecting to WebSocket /ws ...")
    async with websockets.connect(WS_URL) as ws:
        await ws.send("Plan a 7-day honeymoon trip to Paris.")
        response = await ws.recv()
        data = json.loads(response)
        print(" /ws Response:\n", json.dumps(data, indent=4))


# ---------- Run both ----------
if __name__ == "__main__":
    test_chat_endpoint()
    asyncio.run(test_ws_endpoint())
