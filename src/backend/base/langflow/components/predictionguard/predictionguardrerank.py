from langflow.base.compressors.model import LCCompressorComponent
from langflow.field_typing import BaseDocumentCompressor
from langflow.inputs.inputs import SecretStrInput
from langflow.io import DropdownInput
from langflow.template.field.base import Output


class PredictionGuardRerankComponent(LCCompressorComponent):
    display_name = "PredictionGuard Rerank"
    description = "Rerank documents using the PredictionGuard API."
    name = "PredictionGuardRerank"
    icon = "PredictionGuard"

    inputs = [
        *LCCompressorComponent.inputs,
        SecretStrInput(
            name="api_key",
            display_name="PredictionGuard API Key",
        ),
        DropdownInput(
            name="model",
            display_name="Model",
            options=[
                "bge-reranker-v2-m3"
            ],
            value="bge-reranker-v2-m3",
        ),
    ]

    outputs = [
        Output(
            display_name="Reranked Documents",
            name="reranked_documents",
            method="compress_documents",
        ),
    ]

    def build_compressor(self) -> BaseDocumentCompressor:  # type: ignore[type-var]
        try:
            from langchain_predictionguard import PredictionGuardRerank
        except ImportError as e:
            msg = "Please install langchain-predictionguard to use the PredictionGuard model."
            raise ImportError(msg) from e
        return PredictionGuardRerank(
            predictionguard_api_key=self.api_key,
            model=self.model,
        )
