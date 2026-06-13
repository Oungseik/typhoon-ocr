import os
import subprocess
import sys
import time
import urllib.request
import json
from pathlib import Path as LocalPath

from cog import BasePredictor, Input, Path
from openai import OpenAI

sys.path.insert(0, str(LocalPath(__file__).parent / "packages" / "typhoon_ocr"))

from typhoon_ocr import prepare_ocr_messages


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.port = int(os.getenv("VLLM_PORT", "8000"))
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        self.model_id = os.getenv("TYPHOON_OCR_MODEL_ID", "typhoon-ai/typhoon-ocr-7b")
        self.served_model = os.getenv("TYPHOON_OCR_SERVED_MODEL", "typhoon-ocr")
        self.log_path = LocalPath("/tmp/vllm-typhoon-ocr.log")
        self.log_file = None
        self.server = None

    def _start_server(self) -> None:
        vllm_python = os.getenv("VLLM_PYTHON", "/opt/vllm/bin/python")
        command = [
            vllm_python,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model_id,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--served-model-name",
            self.served_model,
            "--dtype",
            os.getenv("TYPHOON_OCR_DTYPE", "float16"),
            "--max-model-len",
            os.getenv("TYPHOON_OCR_MAX_MODEL_LEN", "8192"),
            "--max-num-seqs",
            os.getenv("TYPHOON_OCR_MAX_NUM_SEQS", "1"),
            "--enforce-eager",
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

    def _ensure_server(self) -> None:
        if self.server is not None and self.server.poll() is None:
            return
        self._start_server()

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
            if self.log_file is not None:
                self.log_file.flush()
            text = self.log_path.read_text(errors="replace")
            return text[-max_chars:]
        except Exception as error:
            return f"Could not read vLLM logs: {error}"

    def _parse_pages(self, pages: str) -> list[int]:
        if not pages.strip():
            return [1]

        parsed_pages = set()
        for part in pages.split(","):
            value = part.strip()
            if not value:
                continue
            if "-" in value:
                start_text, end_text = value.split("-", 1)
                start = int(start_text.strip())
                end = int(end_text.strip())
                if start < 1 or end < start:
                    raise ValueError(f"Invalid page range: {value}")
                parsed_pages.update(range(start, end + 1))
            else:
                page = int(value)
                if page < 1:
                    raise ValueError(f"Invalid page number: {value}")
                parsed_pages.add(page)

        return sorted(parsed_pages) or [1]

    def _run_ocr(
        self,
        file_path: str,
        task_type: str,
        page_number: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> str:
        messages = prepare_ocr_messages(
            pdf_or_image_path=file_path,
            task_type=task_type,
            page_num=page_number,
            figure_language="Thai",
        )
        client = OpenAI(base_url=self.base_url, api_key="no-key")
        response = client.chat.completions.create(
            model=self.served_model,
            messages=messages,
            max_tokens=max_tokens,
            extra_body={
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
            },
        )
        text_output = response.choices[0].message.content
        if task_type == "v1.5":
            return text_output
        return json.loads(text_output)["natural_text"]

    def predict(
        self,
        file: Path = Input(
            description="Image or PDF file to OCR. Supported: PDF, PNG, JPG, JPEG."
        ),
        task_type: str = Input(
            description="OCR prompt mode.",
            default="v1.5",
            choices=["default", "structure", "v1.5"],
        ),
        pages: str = Input(
            description="PDF pages to process, for example 1, 1,3, or 1-3. Images always use page 1.",
            default="",
        ),
        max_tokens: int = Input(
            description="Maximum generated tokens.",
            default=4096,
            ge=1,
            le=8192,
        ),
        temperature: float = Input(
            description="Sampling temperature.",
            default=0.1,
            ge=0.0,
            le=2.0,
        ),
        top_p: float = Input(
            description="Nucleus sampling top-p value.",
            default=0.6,
            ge=0.0,
            le=1.0,
        ),
        repetition_penalty: float = Input(
            description="Repetition penalty.",
            default=1.2,
            ge=0.0,
            le=2.0,
        ),
    ) -> str:
        self._ensure_server()

        file_path = str(file)
        page_numbers = [1]
        if LocalPath(file_path).suffix.lower() == ".pdf":
            page_numbers = self._parse_pages(pages)

        results = [
            (
                page_number,
                self._run_ocr(
                    file_path=file_path,
                    task_type=task_type,
                    page_number=page_number,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                ),
            )
            for page_number in page_numbers
        ]

        if len(results) == 1:
            return results[0][1]

        return "\n\n---\n\n".join(
            f"## Page {page_number}\n\n{text}" for page_number, text in results
        )
