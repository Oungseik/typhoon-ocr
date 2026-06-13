## Typhoon OCR

Typhoon OCR is a model for extracting structured markdown from images or PDFs. It supports document layout analysis and table extraction, returning results in markdown or HTML. This package is a simple Gradio website to demonstrate the performance of Typhoon OCR.

### Features

- Upload a PDF or image (single page)
- Extracts and reconstructs document content as markdown
- Supports different prompt modes for layout or structure
- Language: English, Thai
- Uses a local or remote OpenAI-compatible API (e.g., vllm, opentyphoon.ai)
- See blog for more detail https://opentyphoon.ai/blog/en/typhoon-ocr-release

### Requirements

- Linux / Mac with python (window not supported at the moment)

### Install

```bash
pip install typhoon-ocr
```

or to run the gradio app.

```bash
pip install -r requirements.txt
# edit .env
# pip install vllm # optional for hosting a local server
```

### Mac specific

```
brew install poppler
# The following binaries are required and provided by poppler:
# - pdfinfo
# - pdftoppm
```

### Linux specific

```
sudo apt-get update
sudo apt-get install poppler-utils
# The following binaries are required and provided by poppler-utils:
# - pdfinfo
# - pdftoppm
```

### Start vllm

```bash
vllm serve scb10x/typhoon-ocr-7b --served-model-name typhoon-ocr --dtype bfloat16 --port 8101
```

### Run Gradio demo

```bash
python app.py
```

### Replicate deployment

This fork includes a Replicate/Cog wrapper in `predict.py`. The wrapper starts the
local vLLM server and exposes a playground-style form with:

- `file`: image or PDF input
- `task_type`: `default`, `structure`, or `v1.5`
- `pages`: PDF page selection such as `1`, `1,3`, or `1-3`
- `max_tokens`, `temperature`, `top_p`, `repetition_penalty`: generation settings

Use `examples/invoice.png` as a sample image when testing the Replicate model page.

To deploy without local Docker, add a GitHub repository secret named
`REPLICATE_API_TOKEN` with the token from `https://replicate.com/auth/token`.
Pushing deployment-related files to `master` runs the `Deploy to Replicate`
GitHub Action, or you can trigger it manually from the Actions tab. The default
target is `r8.im/oungseik/typhoon-ocr`.

For 16 GB GPUs such as T4, the wrapper uses lower-memory vLLM defaults:

- `TYPHOON_OCR_MAX_MODEL_LEN=4096`
- `TYPHOON_OCR_GPU_MEMORY_UTILIZATION=0.80`
- `TYPHOON_OCR_MAX_NUM_SEQS=1`

You can override these in the model version environment. If startup still fails
with CUDA OOM while loading weights, set `TYPHOON_OCR_CPU_OFFLOAD_GB=2` or use a
larger Replicate GPU.

The same settings are also exposed as advanced prediction inputs:

- `vllm_max_model_len`
- `vllm_gpu_memory_utilization`
- `vllm_cpu_offload_gb`
- `vllm_extra_args`

After this version is pushed once, use those inputs from the Replicate playground
or API to tune startup memory without rebuilding the image. vLLM runs as a warm
server inside the worker, so the first prediction in a warm worker fixes these
startup settings until the worker is restarted.

### Dependencies

- openai
- python-dotenv
- ftfy
- pypdf
- gradio
- vllm (for hosting an inference server)
- pillow

### Debug

- If `Error processing document` occur. Make sure you have install `brew install poppler` or `apt-get install poppler-utils`.

### License

This project is licensed under the Apache 2.0 License. See individual datasets and checkpoints for their respective licenses.
