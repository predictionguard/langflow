from predictionguard import PredictionGuard
from langflow.base.io.text import TextComponent
from langflow.io import BoolInput, MultilineInput, Output, SecretStrInput, StrInput
from langflow.schema.message import Message


class PredictionGuardPIIComponent(TextComponent):
    display_name = "PII Guardrail"
    description = "Check text inputs for PII."
    icon = "PredictionGuard"
    name = "PredictionGuardPII"

    inputs = [
        MultilineInput(
            name="input_value",
            display_name="Text",
            info="Text to be checked for PII",
        ),
        SecretStrInput(
            name="api_key",
            display_name="Prediction Guard API Key",
            info="The Prediction Guard API Key to use.",
            advanced=False,
            required=True,
        ),
        BoolInput(
            name="replace",
            display_name="Replace",
            info="Whether to replace PII if it is present."
        ),
        StrInput(
            name="replace_method",
            display_name="PII Replace Method",
            info="What method to replace present PII with. Possible values are 'category', 'fake', 'mask', and 'random'."
        ),
    ]
    outputs = [
        Output(display_name="Message", name="text", method="text_response"),
    ]

    def text_response(self) -> Message:
        prompt = self.input_value
        predictionguard_api_key = self.api_key
        replace = self.replace
        replace_method = self.replace_method

        client = PredictionGuard(
            api_key=predictionguard_api_key,
        )

        res = client.pii.check(
            prompt=prompt,
            replace=replace,
            replace_method=replace_method,
        )
        if "new_prompt" in res["checks"][0].keys():
            checked_text = res["checks"][0]["new_prompt"]
        elif "types_and_positions" in res["checks"][0].keys():
            checked_text = res["checks"][0]["types_and_positions"]

        return Message(
            text=checked_text,
        )