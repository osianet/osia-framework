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
                                    # The Signal REST API returns the raw internal ID in 'groupId'
                                    # To send back to a group, we need: group.base64(groupId)
                                    raw_group_id = group_info.get("groupId")
                                    if raw_group_id:
                                        if raw_group_id.startswith("group."):
                                            group_id = raw_group_id
                                        else:
                                            # Encode to base64 and add prefix
                                            encoded_id = base64.b64encode(raw_group_id.encode()).decode()
                                            group_id = f"group.{encoded_id}"

                            if "dataMessage" in envelope:
                                data_msg = envelope["dataMessage"]
                                text = data_msg.get("message", "")
                                
                                if text:
                                    print(f"[Signal] New message from {source} (Resolved Group: {group_id}): {text}")
                                    
                                    # If group_id is found, it is the source for replies
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
