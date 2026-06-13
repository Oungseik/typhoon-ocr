import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path as LocalPath

from cog import BasePredictor, Input, Path

sys.path.insert(0, str(LocalPath(__file__).parent / "packages" / "typhoon_ocr"))

from typhoon_ocr import ocr_document


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.port = int(os.getenv("VLLM_PORT", "8000"))
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        self.model_id = os.getenv("TYPHOON_OCR_MODEL_ID", "typhoon-ai/typhoon-ocr-7b")
        self.served_model = os.getenv("TYPHOON_OCR_SERVED_MODEL", "typhoon-ocr")
        self.log_path = LocalPath("/tmp/vllm-typhoon-ocr.log")

        command = [
            "vllm",
            "serve",
            self.model_id,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--served-model-name",
            self.served_model,
            "--dtype",
            "bfloat16",
            "--max-model-len",
            os.getenv("TYPHOON_OCR_MAX_MODEL_LEN", "32000"),
            "--trust-remote-code",
        ]

        self.log_file = self.log_path.open("w")
        self.server = subprocess.Popen(
            command,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._wait_for_server()

    def _wait_for_server(self) -> None:
        deadline = time.time() + int(os.getenv("VLLM_STARTUP_TIMEOUT", "1800"))
        url = f"{self.base_url}/models"

        while time.time() < deadline:
            if self.server.poll() is not None:
                raise RuntimeError(
                    "vLLM exited before it became ready. "
                    f"Last logs:\n{self._tail_logs()}"
                )

            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    if response.status == 200:
                        return
            except Exception:
                time.sleep(5)

        self.server.terminate()
        raise TimeoutError(
            "Timed out waiting for vLLM to start. "
            f"Last logs:\n{self._tail_logs()}"
        )

    def _tail_logs(self, max_chars: int = 6000) -> str:
        try:
            self.log_file.flush()
            text = self.log_path.read_text(errors="replace")
            return text[-max_chars:]
        except Exception as error:
            return f"Could not read vLLM logs: {error}"

    def predict(
        self,
        document: Path = Input(
            description="PDF or image file to OCR. Supported: PDF, PNG, JPG, JPEG."
        ),
        task_type: str = Input(
            description="OCR prompt mode.",
            default="default",
            choices=["default", "structure", "v1.5"],
        ),
        page_number: int = Input(
            description="1-indexed page number for PDFs. Images always use page 1.",
            default=1,
        ),
        target_image_dim: int = Input(
            description="Longest rendered image dimension in pixels.",
            default=1800,
        ),
        target_text_length: int = Input(
            description="Maximum PDF anchor text length for default/structure modes.",
            default=8000,
        ),
        figure_language: str = Input(
            description="Language for figure descriptions in v1.5 mode.",
            default="Thai",
            choices=["Thai", "English"],
        ),
    ) -> str:
        return ocr_document(
            str(document),
            task_type=task_type,
            target_image_dim=target_image_dim,
            target_text_length=target_text_length,
            page_num=page_number,
            base_url=self.base_url,
            api_key="no-key",
            model=self.served_model,
            figure_language=figure_language,
        )
