import yaml
from pathlib import Path
from pydantic import BaseModel


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8642


class DatabaseConfig(BaseModel):
    path: str = "llm_router.db"


class ProviderConfig(BaseModel):
    name: str
    base_url: str
    api_key_env: str
    daily_reset_utc_hour: int = 0
    account_id_env: str | None = None


class Config(BaseModel):
    server: ServerConfig = ServerConfig()
    database: DatabaseConfig = DatabaseConfig()
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
