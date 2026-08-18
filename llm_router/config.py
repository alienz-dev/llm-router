import yaml
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

# Load .env into os.environ before any os.getenv() calls
load_dotenv(Path(__file__).parent.parent / ".env")


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8642


class DatabaseConfig(BaseModel):
    path: str = "llm_router.db"


class SchedulerConfig(BaseModel):
    batch_window_start: str = "14:00"
    batch_window_end: str = "20:00"
    discovery_interval_hours: int = 6


class ProviderConfig(BaseModel):
    name: str
    base_url: str
    api_key_env: str
    daily_reset_utc_hour: int = 0
    account_id_env: str | None = None
    priority: float = 1.0
    timeout: int = 30
    # Narrower than routing.passthrough_params for a provider that 400s on
    # unknown fields. None means "use the global list".
    passthrough_params: list[str] | None = None


class RoutingConfig(BaseModel):
    """Which caller-supplied parameters reach a provider.

    This list is the kill switch for tool calling and every other forwarded
    field: shorten it to `[max_tokens, temperature, response_format]` and the
    router is back to its pre-tools behaviour in one edit, with no code change.
    """

    passthrough_params: list[str] = [
        "max_tokens", "temperature", "response_format", "tools", "tool_choice",
        "parallel_tool_calls", "stream_options", "stop", "seed", "top_p", "n", "user",
    ]


class Config(BaseModel):
    server: ServerConfig = ServerConfig()
    database: DatabaseConfig = DatabaseConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    routing: RoutingConfig = RoutingConfig()
    providers: dict[str, ProviderConfig] = {}


_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is not None:
        return _config

    config_path = Path("config.yaml")
    if not config_path.exists():
        _config = Config()
        return _config

    with open(config_path) as f:
        data = yaml.safe_load(f)

    _config = Config(**data)
    return _config
