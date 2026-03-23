import asyncio
import websockets
import json
import os
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
                                    # The Signal REST API returns the string we need for replies in the 'groupId' field.
                                    # Based on debug logs, it is e.g. "pugWDcTiDcjExyOKVfJ6blbYxOPPNifdqTTD37jnwMo="
                                    # We just need to prefix it with "group."
                                    group_id = group_info.get("groupId")
                                    if group_id and not group_id.startswith("group."):
                                        group_id = f"group.{group_id}"

                            if "dataMessage" in envelope:
                                data_msg = envelope["dataMessage"]
                                text = data_msg.get("message", "")
                                
                                if text:
                                    print(f"[Signal] New message from {source} (Resolved Group: {group_id}): {text}")
                                    
                                    # If it's a group message, the source for the orchestrator reply is the group ID
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
