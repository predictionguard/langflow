import logging
from typing import TYPE_CHECKING

from langflow.base.data.utils import IMG_FILE_TYPES
from langflow.custom import Component
from langflow.io import FileInput, HandleInput, MessageInput, Output
from langflow.schema import Data

if TYPE_CHECKING:
    from langflow.field_typing import Embeddings
    from langflow.schema.image import Image


class ImageEmbedderComponent(Component):
    display_name: str = "Image Embedder"
    description: str = "Generate embeddings for a given images using the specified embedding model."
    icon = "PredictionGuard"
    inputs = [
        HandleInput(
            name="embedding_model",
            display_name="Embedding Model",
            info="The embedding model to use for generating embeddings.",
            input_types=["Embeddings"],
            required=True,
        ),
        FileInput(
            name="image",
            display_name="Image",
            file_types=IMG_FILE_TYPES,
            info="The image to embed.",
            temp_file=True,
        ),
    ]
    outputs = [
        Output(display_name="Embedding Data", name="embeddings", method="generate_embeddings"),
    ]

    def generate_embeddings(self) -> Data:
        try:
            embedding_model: Embeddings = self.embedding_model
            image: Image = self.image

            # Combine validation checks to reduce nesting
            if not embedding_model or not hasattr(embedding_model, "embed_documents"):
                msg = "Invalid or incompatible embedding model"
                raise ValueError(msg)

            embeddings = embedding_model.embed_images([image])
            if not embeddings or not isinstance(embeddings, list):
                msg = "Invalid embeddings generated"
                raise ValueError(msg)

            embedding_vector = embeddings[0]
            self.status = {"image": image, "embeddings": embedding_vector}
            return Data(data={"image": image, "embeddings": embedding_vector})
        except Exception as e:
            logging.exception("Error generating embeddings")
            error_data = Data(data={"image": "", "embeddings": [], "error": str(e)})
            self.status = {"error": str(e)}
            return error_data