import asyncio
import logging
import os
from dotenv import load_dotenv
from src.orchestrator import OsiaOrchestrator


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


async def main():
    load_dotenv()
    setup_logging()

    logger = logging.getLogger("osia")
    logger.info("Initializing OSIA Framework...")

    orchestrator = OsiaOrchestrator()
    try:
        await orchestrator.run_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down OSIA Orchestrator...")
    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
