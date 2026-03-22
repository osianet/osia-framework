import asyncio
import os
from dotenv import load_dotenv
from src.orchestrator import OsiaOrchestrator

async def main():
    # Load environment variables
    load_dotenv()
    
    print("Initializing OSIA Framework...")
    
    # Initialize and run the Orchestrator
    orchestrator = OsiaOrchestrator()
    
    try:
        await orchestrator.run_forever()
    except KeyboardInterrupt:
        print("\nShutting down OSIA Orchestrator...")

if __name__ == "__main__":
    asyncio.run(main())
