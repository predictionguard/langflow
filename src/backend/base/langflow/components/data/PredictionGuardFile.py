from defusedxml import ElementTree
from langflow.inputs import StrInput

from predictionguard import PredictionGuard
from langflow.base.data import BaseFileComponent
from langflow.base.data.utils import TEXT_FILE_TYPES, parallel_load_data, read_text_file
from langflow.io import BoolInput, IntInput, SecretStrInput
from langflow.schema import Data
import yaml


class PredictionGuardFileComponent(BaseFileComponent):
    """Handles loading and processing of individual or zipped text files.

    This component supports processing multiple valid files within a zip archive,
    resolving paths, validating file types, and optionally using multithreading for processing.
    """

    display_name = "Prediction Guard File"
    description = "Load a file to be used in your project."
    icon = "PredictionGuard"
    name = "Prediction Guard File"

    VALID_EXTENSIONS = TEXT_FILE_TYPES

    inputs = [
        *BaseFileComponent._base_inputs,
        SecretStrInput(
            name="predictionguard_api_key",
            display_name="PredictionGuard API Key",
            value="PREDICTIONGUARD_API_KEY",
            required=True
        ),
        BoolInput(
            name="embed_images",
            display_name="Embed images",
            advanced=True
        ),
        StrInput(
            name="output_format",
            display_name="Output format",
            advanced=True
        ),
        BoolInput(
            name="chunk_document",
            display_name="Chunk document",
            advanced=True
        ),
        IntInput(
            name="chunk_size",
            display_name="Chunk size",
            advanced=True
        ),
        BoolInput(
            name="enable_OCR",
            display_name="Enable OCR",
            advanced=True
        ),
        BoolInput(
            name="prompt_injection",
            display_name="Prompt Injection",
            advanced=True
        ),
        StrInput(
            name="pii",
            display_name="PII",
            advanced=True
        ),
        StrInput(
            name="replace_method",
            display_name="Replace Method",
            advanced=True
        ),
        BoolInput(
            name="toxicity",
            display_name="Toxicity",
            advanced=True
        ),
        BoolInput(
            name="use_multithreading",
            display_name="[Deprecated] Use Multithreading",
            advanced=True,
            value=True,
            info="Set 'Processing Concurrency' greater than 1 to enable multithreading.",
        ),
        IntInput(
            name="concurrency_multithreading",
            display_name="Processing Concurrency",
            advanced=True,
            info="When multiple files are being processed, the number of files to process concurrently.",
            value=1,
        ),
    ]

    outputs = [
        *BaseFileComponent._base_outputs,
    ]

    def process_files(self, file_list: list[BaseFileComponent.BaseFile]) -> list[BaseFileComponent.BaseFile]:
        """Processes files either sequentially or in parallel, depending on concurrency settings.

        Args:
            file_list (list[BaseFileComponent.BaseFile]): List of files to process.

        Returns:
            list[BaseFileComponent.BaseFile]: Updated list of files with merged data.
        """

        def parse_text_file_to_data(file_path: str, *, silent_errors: bool) -> Data | None:
            try:
                try:
                    client = PredictionGuard(
                        api_key=self.predictionguard_api_key,
                    )
                    response = client.documents.extract.create(
                        file=file_path,
                        embed_images=self.embed_images,
                        output_format=self.output_format,
                        chunk_document=self.chunk_document,
                        chunk_size=self.chunk_size,
                        toxicity=self.toxicity,
                        pii=self.pii,
                        replace_method=self.replace_method,
                        injection=self.prompt_injection,
                    )

                    text = response["contents"]

                except ValueError:
                    text = read_text_file(file_path)

                if file_path.endswith((".yaml", ".yml")):
                    text = yaml.safe_load(text)
                elif file_path.endswith(".xml"):
                    xml_element = ElementTree.fromstring(text)
                    text = ElementTree.tostring(xml_element, encoding="unicode")
            except Exception as e:
                if not silent_errors:
                    msg = f"Error loading file {file_path}: {e}"
                    raise ValueError(msg) from e
                return None

            return Data(data={"file_path": file_path, "text": text})

        def process_file(file_path: str, *, silent_errors: bool = False) -> Data | None:
            """Processes a single file and returns its Data object."""
            try:
                return parse_text_file_to_data(file_path, silent_errors=silent_errors)
            except FileNotFoundError as e:
                msg = f"File not found: {file_path}. Error: {e}"
                self.log(msg)
                if not silent_errors:
                    raise
                return None
            except Exception as e:
                msg = f"Unexpected error processing {file_path}: {e}"
                self.log(msg)
                if not silent_errors:
                    raise
                return None

        if not file_list:
            msg = "No files to process."
            raise ValueError(msg)

        concurrency = 1 if not self.use_multithreading else max(1, self.concurrency_multithreading)
        file_count = len(file_list)

        parallel_processing_threshold = 2
        if concurrency < parallel_processing_threshold or file_count < parallel_processing_threshold:
            if file_count > 1:
                self.log(f"Processing {file_count} files sequentially.")
            processed_data = [process_file(str(file.path), silent_errors=self.silent_errors) for file in file_list]
        else:
            self.log(f"Starting parallel processing of {file_count} files with concurrency: {concurrency}.")
            file_paths = [str(file.path) for file in file_list]
            processed_data = parallel_load_data(
                file_paths,
                silent_errors=self.silent_errors,
                load_function=process_file,
                max_concurrency=concurrency,
            )

        # Use rollup_basefile_data to merge processed data with BaseFile objects
        return self.rollup_data(file_list, processed_data)