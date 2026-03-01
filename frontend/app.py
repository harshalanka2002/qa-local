import os
import time
import requests
import gradio as gr

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:9044").strip()


def qa_answer(context, question):
    context = (context or "").strip()
    question = (question or "").strip()

    if not context:
        return "Please paste some context/passage first.", ""
    if not question:
        return "Please type a question.", ""

    t0 = time.time()
    try:
        r = requests.post(
            f"{BACKEND_URL}/qa",
            json={"context": context, "question": question},
            timeout=120,  # local model can be slower
        )
        r.raise_for_status()
        data = r.json()
        dt = time.time() - t0

        answer = data.get("answer", "")
        meta = data.get("meta", {})
        meta_line = (
            f"Mode: {meta.get('mode','local')} | Model: {meta.get('model','')} | "
            f"Backend time: {meta.get('time_sec','?')}s | Roundtrip: {dt:.2f}s"
        )
        if "score" in meta:
            meta_line += f" | Score: {meta['score']:.3f}"

        return answer, meta_line

    except Exception as e:
        return f"Frontend error: {e}", "Request failed"


with gr.Blocks() as demo:
    gr.Markdown("# ❓ Local Question Answering Assistant (Local mode)")
    gr.Markdown(
        f"Frontend (Gradio) calling backend: `{BACKEND_URL}`\n\n"
        "Paste a passage and ask a question. Backend runs Transformers pipeline locally."
    )

    context = gr.Textbox(label="Context / Passage", lines=10)
    question = gr.Textbox(label="Question", lines=2)

    btn = gr.Button("Get Answer")

    answer_box = gr.Textbox(label="Answer", lines=4, interactive=False)
    meta_box = gr.Textbox(label="Run info", interactive=False)

    btn.click(qa_answer, [context, question], [answer_box, meta_box])

demo.launch(server_name="0.0.0.0", server_port=7044)