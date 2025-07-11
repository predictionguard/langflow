from langchain_openai import ChatOpenAI
from pydantic.v1 import SecretStr

from langflow.base.models.model import LCModelComponent
from langflow.field_typing import LanguageModel
from langflow.field_typing.range_spec import RangeSpec
from langflow.inputs import (
    BoolInput,
    DictInput,
    IntInput,
    SecretStrInput,
    SliderInput,
    StrInput,
)

PG_MODELS = ["Hermes-3-Llama-3.1-70B", "Hermes-3-Llama-3.1-8B"]


class PredictionGuardComponent(LCModelComponent):
    display_name = "Prediction Guard"
    description = "Generates text using Prediction Guard LLMs."
    icon = "PredictionGuard"
    name = "PredictionGuardModel"

    inputs = [
        *LCModelComponent._base_inputs,
        IntInput(
            name="max_tokens",
            display_name="Max Tokens",
            advanced=True,
            info="The maximum number of tokens to generate. Set to 0 for unlimited tokens.",
            range_spec=RangeSpec(min=0, max=128000),
        ),
        DictInput(
            name="model_kwargs",
            display_name="Model Kwargs",
            advanced=True,
            info="Additional keyword arguments to pass to the model.",
        ),
        BoolInput(
            name="json_mode",
            display_name="JSON Mode",
            advanced=True,
            info="If True, it will output JSON regardless of passing a schema.",
        ),
        StrInput(
            name="model_name",
            display_name="Model Name",
            info="The model to use in the chat request. See more at docs.predictionguard.com.",
            advanced=False,
            value="Hermes-3-Llama-3.1-70B",
            required=True,
        ),
        SecretStrInput(
            name="api_key",
            display_name="Prediction Guard API Key",
            info="The Prediction Guard API Key to use.",
            advanced=False,
            required=True,
        ),
        SliderInput(
            name="temperature",
            display_name="Temperature",
            value=0.1,
            range_spec=RangeSpec(min=0, max=1, step=0.01),
        ),
        IntInput(
            name="seed",
            display_name="Seed",
            info="The seed controls the reproducibility of the job.",
            advanced=True,
            value=1,
        ),
        IntInput(
            name="max_retries",
            display_name="Max Retries",
            info="The maximum number of retries to make when generating.",
            advanced=True,
            value=5,
        ),
        IntInput(
            name="timeout",
            display_name="Timeout",
            info="The timeout for requests to OpenAI completion API.",
            advanced=True,
            value=700,
        ),
    ]

    def build_model(self) -> LanguageModel:
        openai_api_key = self.api_key
        temperature = self.temperature
        model_name = self.model_name
        max_tokens = self.max_tokens
        model_kwargs = self.model_kwargs or {}
        openai_api_base = "https://api.predictionguard.com"
        json_mode = self.json_mode
        seed = self.seed
        max_retries = self.max_retries
        timeout = self.timeout

        api_key = (
            SecretStr(openai_api_key).get_secret_value() if openai_api_key else None
        )
        output = ChatOpenAI(
            max_tokens=max_tokens or None,
            model_kwargs=model_kwargs,
            model=model_name,
            base_url=openai_api_base,
            api_key=api_key,
            temperature=temperature if temperature is not None else 0.1,
            seed=seed,
            max_retries=max_retries,
            request_timeout=timeout,
        )
        if json_mode:
            output = output.bind(response_format={"type": "json_object"})

        return output

    def _get_exception_message(self, e: Exception):
        try:
            from openai import BadRequestError
        except ImportError:
            return None
        if isinstance(e, BadRequestError):
            message = e.body.get("message")
            if message:
                return message
        return None
