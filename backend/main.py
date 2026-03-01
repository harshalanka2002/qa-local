import time
from typing import Optional, Dict, Any
import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

MODEL_ID = "deepset/bert-base-uncased-squad2"

app = FastAPI(title="qa-local backend", version="1.0")

# Load once, reuse (saves time)
qa_pipe = None
pipe_lock = threading.Lock()


class QARequest(BaseModel):
    context: str
    question: str


class QAResponse(BaseModel):
    answer: str
    meta: Dict[str, Any]


def get_pipe():
    global qa_pipe
    if qa_pipe is None:
        with pipe_lock:
            if qa_pipe is None:
                qa_pipe = pipeline("question-answering", model=MODEL_ID)
    return qa_pipe


@app.get("/health")
def health():
    return {"ok": True, "mode": "local", "model": MODEL_ID}


@app.post("/qa", response_model=QAResponse)
def qa(req: QARequest):
    context = (req.context or "").strip()
    question = (req.question or "").strip()

    if not context:
        raise HTTPException(status_code=400, detail="Context is required.")
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    if len(context) > 8000:
        context = context[:8000] + "..."

    t0 = time.time()
    try:
        qa = get_pipe()
        out = qa(question=question, context=context)
        dt = time.time() - t0

        answer = out.get("answer", "") or "(No answer found in the provided context.)"
        score = out.get("score", None)

        meta = {
            "mode": "local",
            "model": MODEL_ID,
            "time_sec": round(dt, 4),
        }
        if score is not None:
            meta["score"] = float(score)

        return {"answer": answer, "meta": meta}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Local error: {e}")