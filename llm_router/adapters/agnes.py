from .base import OpenAICompatibleAdapter


class AgnesAdapter(OpenAICompatibleAdapter):
    provider_name = "agnes"

    def __init__(self, api_key: str):
        super().__init__(api_key, base_url="https://apihub.agnes-ai.com/v1")
