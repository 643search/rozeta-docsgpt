import logging
from application.llm.base import BaseLLM
from application.core.settings import settings
from application.storage.storage_creator import StorageCreator

logger = logging.getLogger(__name__)


class AnthropicLLM(BaseLLM):
    def __init__(self, api_key=None, user_api_key=None, base_url=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from anthropic import Anthropic
        self.api_key = api_key or settings.ANTHROPIC_API_KEY or settings.API_KEY
        self.user_api_key = user_api_key
        if base_url:
            self.client = Anthropic(api_key=self.api_key, base_url=base_url)
        else:
            self.client = Anthropic(api_key=self.api_key)
        self.storage = StorageCreator.get_storage()

    def _raw_gen(self, baseself, model, messages, stream=False, tools=None, max_tokens=2048, **kwargs):
        # Convert messages: extract system prompt separately
        system_content = None
        user_messages = []
        for m in messages:
            if m.get("role") == "system":
                system_content = m["content"]
            else:
                user_messages.append(m)
        if not user_messages:
            user_messages = [{"role": "user", "content": "Hello"}]

        kwargs_clean = {k: v for k, v in kwargs.items() if k in ("temperature", "top_p")}
        create_kwargs = {"model": model, "max_tokens": max_tokens, "messages": user_messages, **kwargs_clean}
        if system_content:
            create_kwargs["system"] = system_content

        if stream:
            return self._raw_gen_stream(baseself, model, messages, stream=True, max_tokens=max_tokens, **kwargs)

        response = self.client.messages.create(**create_kwargs)
        return response.content[0].text

    def _raw_gen_stream(self, baseself, model, messages, stream=True, tools=None, max_tokens=2048, **kwargs):
        system_content = None
        user_messages = []
        for m in messages:
            if m.get("role") == "system":
                system_content = m["content"]
            else:
                user_messages.append(m)
        if not user_messages:
            user_messages = [{"role": "user", "content": "Hello"}]

        kwargs_clean = {k: v for k, v in kwargs.items() if k in ("temperature", "top_p")}
        create_kwargs = {"model": model, "max_tokens": max_tokens, "messages": user_messages, **kwargs_clean}
        if system_content:
            create_kwargs["system"] = system_content

        with self.client.messages.stream(**create_kwargs) as stream_obj:
            for text in stream_obj.text_stream:
                yield text
