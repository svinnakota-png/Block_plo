import os
import logging
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from groq import APIConnectionError, APIStatusError, Groq
from pydantic import BaseModel, Field
from starlette.responses import FileResponse

logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).with_name(".env"))

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY not found. Check that .env is beside main.py."
    )

model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
client = Groq(api_key=api_key, timeout=30.0)
app = FastAPI(title="My AI Assistant")

STATIC_DIR = Path(__file__).with_name("static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

SYSTEM_INSTRUCTION = (
    "You are BlackPlo AI, a friendly and helpful general-purpose assistant. "
    "Answer the user's actual question directly and accurately. Be natural "
    "and concise, using clear steps or examples when they help. If you are "
    "uncertain or missing important details, say so or ask a focused "
    "clarifying question instead of making up an answer. Do not claim to "
    "have done anything you have not done."
)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, str]:
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
    messages.extend(
        {"role": item.role, "content": item.content}
        for item in request.history[-12:]
    )
    messages.append({"role": "user", "content": request.message})

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=1,
            max_completion_tokens=2048,
            top_p=1,
            reasoning_effort="medium",
            stream=False,
            stop=None,
        )
    except APIStatusError as exc:
        logger.exception("Groq API request failed")
        raise HTTPException(
            status_code=502,
            detail=f"Groq API error ({exc.status_code}): {exc.message}",
        ) from exc
    except APIConnectionError as exc:
        logger.exception("Could not reach Groq API")
        raise HTTPException(
            status_code=502,
            detail="Could not reach Groq. Please try again shortly.",
        ) from exc

    reply = response.choices[0].message.content if response.choices else None
    if not isinstance(reply, str) or not reply.strip():
        logger.error("Groq returned no text for the chat request")
        raise HTTPException(
            status_code=502,
            detail="Groq returned no text. Please try rephrasing your question.",
        )
    return {"reply": reply}
