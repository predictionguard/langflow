from predictionguard import PredictionGuard
from langflow.field_typing.range_spec import RangeSpec
from langflow.base.io.text import TextComponent
from langflow.io import MultilineInput, SecretStrInput, FloatInput, Output
from langflow.schema.message import Message


class PredictionGuardInjectionComponent(TextComponent):
    display_name = "Prompt Injection Guardrail"
    description = "Check text inputs for Prompt Injection."
    icon = "PredictionGuard"
    name = "PredictionGuardInjection"

    inputs = [
        MultilineInput(
            name="input_value",
            display_name="Text",
            info="Text to be checked for prompt injection.",
        ),
        SecretStrInput(
            name="api_key",
            display_name="Prediction Guard API Key",
            info="The Prediction Guard API Key to use.",
            required=True,
        ),
        FloatInput(
            name="threshold",
            display_name="Threshold",
            info="The threshold to block prompt injections at.",
            range_spec=RangeSpec(min=0.0, max=1.0),
        ),
    ]
    outputs = [
        Output(display_name="Message", name="text", method="text_response"),
    ]

    def text_response(self) -> Message:
        prompt = self.input_value
        predictionguard_api_key = self.api_key
        threshold = self.threshold

        client = PredictionGuard(
            api_key=predictionguard_api_key,
        )

        res = client.injection.check(
            prompt=prompt,
        )

        if res["checks"][0]["probability"] < threshold:
            checked_text = prompt
        elif res["checks"][0]["probability"] >= threshold:
            msg = "error: prompt injection detected."
            raise ValueError(msg)

        return Message(
            text=checked_text,
        )