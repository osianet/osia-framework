import asyncio
import websockets
import json
import os
import base64
import redis.asyncio as redis
from dotenv import load_dotenv

load_dotenv()

SIGNAL_NUMBER = os.getenv("SIGNAL_SENDER_NUMBER")
if not SIGNAL_NUMBER:
    raise ValueError("SIGNAL_SENDER_NUMBER environment variable is required")

_signal_ws_base = os.getenv("SIGNAL_WS_URL", "ws://localhost:8081")
SIGNAL_WS_URL = f"{_signal_ws_base}/v1/receive/{SIGNAL_NUMBER}"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
TASK_QUEUE = os.getenv("OSIA_TASK_QUEUE", "osia:task_queue")

async def listen_to_signal():
    """Connects to the Signal REST API WebSocket and listens for incoming messages."""
    redis_client = redis.from_url(REDIS_URL)
    
    print(f"Signal Gateway starting... Listening on {SIGNAL_WS_URL}")
    
    while True:
        try:
            async with websockets.connect(SIGNAL_WS_URL) as websocket:
                print("Connected to Signal WebSocket!")
                while True:
                    message_str = await websocket.recv()
                    try:
                        data = json.loads(message_str)
                        
                        if "envelope" in data:
                            envelope = data["envelope"]
                            source = envelope.get("source")
                            
                            group_id = None
                            if "dataMessage" in envelope:
                                group_info = envelope["dataMessage"].get("groupInfo")
                                if group_info:
                                    raw_group_id = group_info.get("groupId")
                                    if raw_group_id:
                                        if raw_group_id.startswith("group."):
                                            group_id = raw_group_id
                                        else:
                                            # The Signal REST API often returns the internal base64 ID
                                            # We need to ensure it has the group. prefix
                                            # And if it's the raw ID, it needs to be base64 encoded
                                            # Based on logs, pugWDcTi... is the internal ID.
                                            # group.cHVnV0Rj... is the public ID.
                                            # cHVnV0Rj... is base64(pugWDcTi...)
                                            encoded_id = base64.b64encode(raw_group_id.encode()).decode()
                                            group_id = f"group.{encoded_id}"

                            if "dataMessage" in envelope:
                                data_msg = envelope["dataMessage"]
                                text = data_msg.get("message", "")
                                
                                if text:
                                    print(f"[Signal] New message from {source} (Resolved Group: {group_id}): {text}")
                                    
                                    task_source = f"signal:{group_id}" if group_id else f"signal:{source}"
                                    
                                    task = {
                                        "source": task_source,
                                        "query": text
                                    }
                                    
                                    await redis_client.rpush(TASK_QUEUE, json.dumps(task))
                                    print(f"[*] Task pushed to {TASK_QUEUE}")
                                    
                    except json.JSONDecodeError:
                        pass
                        
        except Exception as e:
            print(f"WebSocket connection dropped: {e}. Reconnecting in 5 seconds...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(listen_to_signal())
