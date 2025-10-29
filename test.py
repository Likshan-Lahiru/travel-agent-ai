import asyncio
import json
import websockets

WS_URL = "wss://travel-agent-ai.onrender.com/ws"

async def get_once():
    print("🌐 Connecting…")
    async with websockets.connect(WS_URL) as ws:
        msg = await ws.recv()           # server pushes one payload
        print(" /ws Response:\n", json.dumps(json.loads(msg), indent=2))

if __name__ == "__main__":
    asyncio.run(get_once())


