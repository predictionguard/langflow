from langchain_predictionguard import PredictionGuardEmbeddings
from langflow.base.embeddings.model import LCEmbeddingsModel
from langflow.field_typing import Embeddings
from langflow.io import DropdownInput, StrInput, SecretStrInput


class PredictionGuardEmbeddingsComponent(LCEmbeddingsModel):
    display_name = "Prediction Guard Embeddings"
    description = "Generate embeddings using Prediction Guard hosted models."
    icon = "PredictionGuard"
    name = "PredictionGuardEmbeddings"

    inputs = [
        DropdownInput(
            name="model",
            display_name="Model",
            advanced=False,
            options=["bge-m3", "bridgetower-large-itm-mlm-itc", "multilingual-e5-large-instruct"],
            value="bge-m3",
        ),
        SecretStrInput(
            name="predictionguard_api_key",
            display_name="PredictionGuard API Key",
            value="PREDICTIONGUARD_API_KEY",
            required=True
        ),
    ]

    def build_embeddings(self) -> Embeddings:
        return PredictionGuardEmbeddings(
            model=self.model,
            predictionguard_api_key=self.predictionguard_api_key or None,
        )