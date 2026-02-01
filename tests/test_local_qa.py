import pytest
from transformers import pipeline

MODEL_ID = "deepset/bert-base-uncased-squad2"

@pytest.fixture(scope="session")
def qa():
    # Loads once for all tests (cached in runner once downloaded)
    return pipeline("question-answering", model=MODEL_ID)

def test_answer_contains_expected_phrase(qa):
    context = (
        "Gravity is a force that attracts two bodies toward each other. "
        "Newton described it as proportional to the product of their masses "
        "and inversely proportional to the square of the distance between them."
    )
    question = "What is gravity proportional to?"
    out = qa(question=question, context=context)

    assert "answer" in out
    assert isinstance(out["answer"], str)
    assert len(out["answer"].strip()) > 0

    # Loose check (model wording may vary slightly)
    assert "product" in out["answer"].lower()
    assert "mass" in out["answer"].lower()

def test_no_answer_case_returns_empty_or_low_score(qa):
    context = "Cats are small domesticated animals that purr."
    question = "What is the capital of France?"
    out = qa(question=question, context=context)

    assert "score" in out
    assert isinstance(out["score"], float)

    # Usually low confidence if answer isn't in the passage
    assert out["score"] < 0.30
