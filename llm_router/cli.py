import click
import uvicorn
from .config import get_config


@click.group()
def main():
    """LLM Router — Free-Tier LLM Gateway"""
    pass


@main.command()
@click.option("--host", default=None, help="Host to bind to")
@click.option("--port", default=None, type=int, help="Port to bind to")
def serve(host, port):
    """Start the LLM Router server."""
    config = get_config()
    uvicorn.run(
        "llm_router.app:app",
        host=host or config.server.host,
        port=port or config.server.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
