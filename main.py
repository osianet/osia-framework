import asyncio
import logging

from dotenv import load_dotenv

from src.orchestrator import OsiaOrchestrator


class _Noisy3rdPartyFilter(logging.Filter):
    """Drop sub-WARNING records from libraries that spam the logs.

    WeasyPrint's capture_logs() context manager forcibly calls
    logger.setLevel(DEBUG) on the fontTools logger during every PDF render,
    so setLevel()-based suppression is overridden on every document.
    A filter attached to the root handler is the only reliable fix — it
    runs after the logger-level gate and cannot be overridden by setLevel().
    """

    _NOISY_PREFIXES = ("fontTools.",)

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno < logging.WARNING:
            for prefix in self._NOISY_PREFIXES:
                if record.name.startswith(prefix):
                    return False
        return True


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    # Attach a filter to every handler on the root logger so fontTools spam
    # is silenced even when WeasyPrint resets the logger level during renders.
    _filter = _Noisy3rdPartyFilter()
    for handler in logging.root.handlers:
        handler.addFilter(_filter)


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
