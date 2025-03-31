import math
from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class ModelResponses:
    candidates: List[str] = field(default_factory=list)
    failed: bool = False

    @classmethod
    def load_from_openai(cls, response):
        return cls([response.output_text])

    @classmethod
    def load_from_gemini(cls, response):
        try:
            obj = cls([response.text])
        except ValueError:
            obj = cls([c.text for c in response.candidates])
        return obj

    @classmethod
    def load_from_anthropic(cls, anthropic_response):
        text_type = [
            chunk.text
            for chunk in anthropic_response.content
            if chunk.type == "text"
        ]
        return cls([text_type[0]] if text_type else [])

    @classmethod
    def default_failed(cls):
        """
        Returns a default failed response with an error message.
        """
        return cls(
            ["Failed to process the request. Please try again later."],
            failed=True,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


@dataclass
class LLMTask:
    message: Union[str, List[str]] = ""
    max_tokens: float = math.inf
    query_type: str = "chat"
    model: str = "gpt-4o"
    temperature: float = 1.0
    top_p: float = 1
    presence_penalty: float = 0
    frequency_penalty: float = 0
    thinking_budget: int = 2048
    system_prompt: Optional[str] = None
    stop: Optional[str] = None
    timeout: Optional[float] = None
    n: int = 1
    history: List[dict] = field(default_factory=list)

    def update_conversation(self, response, new_message):
        if not len(self.history) and self.system_prompt is not None:
            history = [{"role": "system", "content": self.system_prompt}]
        elif not len(self.history):
            history = []
        else:
            history = self.history

        history.append({"role": "user", "content": str(self.message)})
        history.append({"role": "assistant", "content": response})

        self.history = history
        self.message = new_message
