import time
import gradio as gr
from transformers import pipeline

MODEL_ID = "deepset/bert-base-uncased-squad2"

qa_pipe = None

def load_model():
    global qa_pipe
    if qa_pipe is None:
        # Local CPU pipeline
        qa_pipe = pipeline("question-answering", model=MODEL_ID)
    return qa_pipe

def answer_local(context, question):
    context = (context or "").strip()
    question = (question or "").strip()

    if not context:
        return "Please paste some context/passage first.", ""
    if not question:
        return "Please type a question.", ""

    # (optional) avoid super huge inputs that slow CPU
    if len(context) > 8000:
        context = context[:8000] + "..."

    t0 = time.time()
    try:
        qa = load_model()
        out = qa(question=question, context=context)
        dt = time.time() - t0

        answer = out.get("answer", "")
        score = out.get("score", None)

        if not answer or answer.strip() == "":
            answer = "(No answer found in the provided context.)"

        meta = f"Mode: Local | Model: {MODEL_ID} | Time: {dt:.2f}s"
        if score is not None:
            meta += f" | Score: {score:.3f}"

        return answer, meta

    except Exception as e:
        return f"Local error: {e}", "Local error"

with gr.Blocks() as demo:
    gr.Markdown("# ❓ Local Question Answering Assistant (CPU)")
    gr.Markdown(
        "Paste a passage (context) and ask a question. "
        "This runs **locally** using an extractive QA model."
    )

    context = gr.Textbox(
        label="Context / Passage",
        lines=10,
        placeholder="Paste paragraph, notes, article excerpt..."
    )
    question = gr.Textbox(
        label="Question",
        lines=2,
        placeholder="Example: What is gravity described as?"
    )

    btn = gr.Button("Get Answer")

    answer_box = gr.Textbox(label="Answer", lines=4, interactive=False)
    meta_box = gr.Textbox(label="Run info", interactive=False)

    btn.click(answer_local, [context, question], [answer_box, meta_box])

demo.launch()
