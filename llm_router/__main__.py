import uvicorn
from .config import get_config


def main():
    config = get_config()
    uvicorn.run(
        "llm_router.app:app",
        host=config.server.host,
        port=config.server.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
