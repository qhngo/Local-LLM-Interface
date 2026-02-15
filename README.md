# Local LLM Interface

FastAPI service that exposes a local `llama-server` model through a stable `/answer` API.

## Overview

- Starts and stops `llama-server` with the API lifecycle.
- Accepts plain-text or JSON-style prompts.
- Queues requests and returns a normalized response.

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (compatible with CUDA 12.4 build)
- Model file: `models/SeaLLM-7B-v2.q8_0.gguf`
- Binary: `tools/llama-b7966-bin-win-cuda-12.4-x64/llama-server.exe`

## Configuration

The API reads `.env` from the repository root.

| Variable | Default | Description |
|---|---|---|
| `WEBAPP_LLM_API_PORT` | `8081` | Port used by FastAPI |

Example `.env`:

```env
WEBAPP_LLM_API_PORT=8081
```

## Run

From the project root:

```bash
pip install -r code/requirements.txt
python code/v1/webapp_llm_api/main.py
```

Server base URL (default): `http://127.0.0.1:8081`

## API

### Health

`GET /`

Response:

```json
{
  "status": "ok",
  "endpoint": "/answer"
}
```

### Answer Endpoint

`/answer` supports `GET` and `POST`.

Input precedence:
- If both `question` and `question_json` are provided, `question_json` is used.
- If neither is provided, an empty prompt is sent to the model.

#### GET with plain text

```bash
curl "http://127.0.0.1:8081/answer?question=What%20is%20the%20capital%20of%20France%3F"
```

#### GET with JSON string (URL-encoded)

```bash
curl "http://127.0.0.1:8081/answer?question_json=%7B%22received%22%3A%22Summarize%20the%20benefits%20of%20local%20LLMs%22%7D"
```

#### POST with plain text

```bash
curl -X POST "http://127.0.0.1:8081/answer" \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"Explain retrieval-augmented generation in simple terms.\"}"
```

#### POST with JSON object

```bash
curl -X POST "http://127.0.0.1:8081/answer" \
  -H "Content-Type: application/json" \
  -d "{\"question_json\":{\"received\":\"Give me 3 startup ideas using local AI.\"}}"
```

### Response Schema

```json
{
  "received": "string | object",
  "input_type": "json | text | empty",
  "answer": "string | null",
  "processing_time_s": "number | null"
}
```

Example:

```json
{
  "received": "What is the capital of France?",
  "input_type": "text",
  "answer": "The capital of France is Paris.",
  "processing_time_s": 1.24
}
```

## Operational Notes

- Model requests are sent to `http://127.0.0.1:8080/v1/chat/completions`.
- Startup fails if the model file or `llama-server.exe` is missing.
- The service includes retry logic for transient model-server failures.

## References

- SeaLLM-7B-v2 (GGUF): https://huggingface.co/SeaLLMs/SeaLLM-7B-v2-gguf
- llama.cpp: https://github.com/ggml-org/llama.cpp