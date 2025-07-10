from predictionguard import PredictionGuard
from langflow.field_typing.range_spec import RangeSpec
from langflow.base.io.text import TextComponent
from langflow.io import FloatInput, MultilineInput, Output, SecretStrInput
from langflow.schema.message import Message


class PredictionGuardToxicityComponent(TextComponent):
    display_name = "Toxicity Guardrail"
    description = "Check text outputs for toxicity."
    icon = "PredictionGuard"
    name = "PredictionGuardToxicity"

    inputs = [
        MultilineInput(
            name="input_value",
            display_name="Text",
            info="Text to be checked for toxicity.",
        ),
        SecretStrInput(
            name="api_key",
            display_name="Prediction Guard API Key",
            info="The Prediction Guard API Key to use.",
            advanced=False,
            required=True,
        ),
        FloatInput(
            name="threshold",
            display_name="Threshold",
            info="The threshold to toxic outputs at.",
            range_spec=RangeSpec(min=0.0, max=1.0),
        ),
    ]
    outputs = [
        Output(display_name="Message", name="text", method="text_response"),
    ]

    def text_response(self) -> Message:
        text = self.input_value
        predictionguard_api_key = self.api_key
        threshold = self.threshold

        client = PredictionGuard(
            api_key=predictionguard_api_key,
        )

        res = client.toxicity.check(text)

        if res["checks"][0]["score"] < threshold:
            checked_text = text
        elif res["checks"][0]["score"] >= threshold:
            msg = "error: toxic output detected."
            raise ValueError(msg)

        return Message(
            text=checked_text,
        )