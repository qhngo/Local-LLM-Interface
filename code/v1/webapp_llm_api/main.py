from __future__ import annotations

import json
import os
import time
from pathlib import Path
from queue import Queue
from threading import Event, Lock, Thread
from typing import Any, Dict, Optional, Union
from uuid import uuid4
import subprocess
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field


class Settings(BaseModel):
    app_name: str = "Answer API"
    app_version: str = "0.1.0"
    port: int = Field(default=8081, ge=1, le=65535)
    llama_server_url: str = "http://127.0.0.1:8080"
    llama_request_timeout_s: float = Field(default=120.0, ge=1.0, le=600.0)
    llama_max_retries: int = Field(default=30, ge=0, le=300)
    llama_retry_delay_s: float = Field(default=1.0, ge=0.1, le=10.0)


def load_settings() -> Settings:
    repo_root = Path(__file__).resolve().parents[3]
    env_path = repo_root / ".env"
    load_dotenv(env_path)

    port_raw = os.getenv("WEBAPP_LLM_API_PORT")
    if port_raw is None:
        return Settings()
    try:
        return Settings(port=int(port_raw))
    except ValueError as exc:
        raise ValueError("WEBAPP_LLM_API_PORT must be an integer") from exc


def start_worker() -> None:
    thread = Thread(target=request_worker, daemon=True)
    thread.start()
    start_llama_server()


def stop_worker() -> None:
    stop_llama_server()


@asynccontextmanager
async def lifespan(_: FastAPI):
    start_worker()
    yield
    stop_worker()


SETTINGS = load_settings()
app = FastAPI(
    title=SETTINGS.app_name,
    version=SETTINGS.app_version,
    lifespan=lifespan,
)
REQUEST_QUEUE: Queue[Dict[str, Any]] = Queue()
REQUEST_RESULTS: Dict[str, Dict[str, Any]] = {}
REQUEST_EVENTS: Dict[str, Event] = {}
REQUEST_LOCK = Lock()
LLAMA_PROCESS: Optional[subprocess.Popen] = None


class AnswerRequest(BaseModel):
    question_json: Optional[Dict[str, Any]] = None
    question: Optional[str] = None


class AnswerResponse(BaseModel):
    received: Union[Dict[str, Any], str]
    input_type: str
    answer: Optional[str] = None
    processing_time_s: Optional[float] = None


def process_request(item: Dict[str, Any]) -> str:
    prompt = _extract_prompt(item.get("received"))
    return _call_llama_server(prompt)


def _extract_prompt(received: Union[Dict[str, Any], str, None]) -> str:
    if received is None:
        return ""
    if isinstance(received, str):
        if received.strip().startswith("{"):
            try:
                parsed = json.loads(received)
                if isinstance(parsed, dict):
                    return _extract_prompt_from_dict(parsed)
            except json.JSONDecodeError:
                pass
        return received
    if isinstance(received, dict):
        return _extract_prompt_from_dict(received)
    return str(received)


def _extract_prompt_from_dict(data: Dict[str, Any]) -> str:
    received = data.get("received")
    if isinstance(received, str):
        return received
    if isinstance(received, dict):
        return json.dumps(received)
    if received is not None:
        return str(received)
    return json.dumps(data)


def _call_llama_server(prompt: str) -> str:
    base_url = SETTINGS.llama_server_url.rstrip("/")
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    last_error: Optional[Exception] = None
    for _ in range(SETTINGS.llama_max_retries + 1):
        try:
            request = Request(url, data=body, headers=headers, method="POST")
            with urlopen(request, timeout=SETTINGS.llama_request_timeout_s) as response:
                data = json.loads(response.read().decode("utf-8"))
            return _extract_llama_answer(data)
        except (URLError, HTTPError, json.JSONDecodeError, TimeoutError) as exc:
            last_error = exc
            if SETTINGS.llama_max_retries == 0:
                break
            time.sleep(SETTINGS.llama_retry_delay_s)

    raise RuntimeError("Failed to get response from llama server") from last_error


def _extract_llama_answer(data: Dict[str, Any]) -> str:
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
            text = first.get("text")
            if isinstance(text, str):
                return text
    return ""


def request_worker() -> None:
    while True:
        item = REQUEST_QUEUE.get()
        try:
            start_time = time.perf_counter()
            answer = process_request(item)
            elapsed_s = round(time.perf_counter() - start_time, 2)
            print(
                f"Processed request {item['id']} in {elapsed_s:.2f}s "
                f"(input_type={item.get('input_type')})"
            )
            with REQUEST_LOCK:
                result = REQUEST_RESULTS.get(item["id"])
                if result is not None:
                    result["answer"] = answer
                    result["processing_time_s"] = elapsed_s
                    REQUEST_EVENTS[item["id"]].set()
        finally:
            REQUEST_QUEUE.task_done()


@app.get("/")
def root() -> Dict[str, str]:
    return {"status": "ok", "endpoint": "/answer"}


@app.get("/answer", response_model=AnswerResponse)
def answer_get(
    question_json: Optional[str] = Query(
        default=None, description="JSON string"
    ),
    question: Optional[str] = Query(
        default=None, description="Plain text question"
    ),
) -> AnswerResponse:
    if question_json:
        request_id = str(uuid4())
        return _enqueue_and_wait(request_id, "json", question_json)
    if question:
        request_id = str(uuid4())
        return _enqueue_and_wait(request_id, "text", question)
    request_id = str(uuid4())
    return _enqueue_and_wait(request_id, "empty", "")


@app.post("/answer", response_model=AnswerResponse)
def answer_post(payload: AnswerRequest) -> AnswerResponse:
    if payload.question_json is not None:
        request_id = str(uuid4())
        return _enqueue_and_wait(request_id, "json", payload.question_json)
    if payload.question is not None:
        request_id = str(uuid4())
        return _enqueue_and_wait(request_id, "text", payload.question)
    request_id = str(uuid4())
    return _enqueue_and_wait(request_id, "empty", "")


def _enqueue_and_wait(
    request_id: str, input_type: str, received: Union[Dict[str, Any], str]
) -> AnswerResponse:
    queue_size = REQUEST_QUEUE.qsize()
    print(f"Queue size before enqueue: {queue_size}")
    event = Event()
    with REQUEST_LOCK:
        REQUEST_RESULTS[request_id] = {
            "received": received,
            "input_type": input_type,
            "answer": None,
            "processing_time_s": None,
        }
        REQUEST_EVENTS[request_id] = event
    REQUEST_QUEUE.put(
        {"id": request_id, "input_type": input_type, "received": received}
    )
    event.wait()
    with REQUEST_LOCK:
        result = REQUEST_RESULTS.pop(request_id)
        REQUEST_EVENTS.pop(request_id, None)
    return AnswerResponse(
        received=result["received"],
        input_type=result["input_type"],
        answer=result["answer"],
        processing_time_s=result.get("processing_time_s"),
    )


def start_llama_server() -> None:
    global LLAMA_PROCESS
    if LLAMA_PROCESS is not None and LLAMA_PROCESS.poll() is None:
        return

    repo_root = Path(__file__).resolve().parents[3]
    llama_server = repo_root / "tools" / "llama-b7966-bin-win-cuda-12.4-x64" / "llama-server.exe"
    model_path = repo_root / "models" / "SeaLLM-7B-v2.q8_0.gguf"

    if not llama_server.exists():
        raise FileNotFoundError(f"llama-server not found: {llama_server}")
    if not model_path.exists():
        raise FileNotFoundError(f"model file not found: {model_path}")

    command = [
        str(llama_server),
        "-m",
        str(model_path),
        "-ngl",
        "99",
        "-c",
        "4096",
        "-t",
        "8",
        "-b",
        "512",
    ]
    LLAMA_PROCESS = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def stop_llama_server() -> None:
    global LLAMA_PROCESS
    if LLAMA_PROCESS is None:
        return
    if LLAMA_PROCESS.poll() is None:
        LLAMA_PROCESS.terminate()
    LLAMA_PROCESS = None


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=SETTINGS.port)
