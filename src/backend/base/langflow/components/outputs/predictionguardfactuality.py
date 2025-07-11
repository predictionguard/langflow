from predictionguard import PredictionGuard

from langflow.base.io.text import TextComponent
from langflow.field_typing.range_spec import RangeSpec
from langflow.io import FloatInput, MultilineInput, Output, SecretStrInput
from langflow.schema.message import Message


class PredictionGuardFactualityComponent(TextComponent):
    display_name = "Factuality Guardrail"
    description = "Check text outputs for factuality."
    icon = "PredictionGuard"
    name = "PredictionGuardFactuality"

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
        MultilineInput(
            name="reference",
            display_name="Reference",
            info="Text to be check input against for factuality.",
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
        reference = self.reference
        threshold = self.threshold

        client = PredictionGuard(
            api_key=predictionguard_api_key,
        )

        res = client.factuality.check(
            reference=reference,
            text=text
        )

        if res["checks"][0]["score"] > threshold:
            checked_text = text
        elif res["checks"][0]["score"] <= threshold:
            msg = "error: factuality check failed."
            raise ValueError(msg)

        return Message(
            text=checked_text,
        )
