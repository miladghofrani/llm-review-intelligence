from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from config import ADAPTER_PATH, CATEGORIES
from device_utils import get_device
from model_loader import load_llm_model, load_tokenizer
from peft_trainer import load_saved_peft_model
from translator import Translator

# ── State shared across requests ───────────────────────────────────────────
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = get_device()
    tokenizer = load_tokenizer()
    base_model = load_llm_model(device)
    model = load_saved_peft_model(device, base_model, ADAPTER_PATH)
    model.eval()

    _state["device"] = device
    _state["tokenizer"] = tokenizer
    _state["model"] = model
    _state["categories_str"] = ", ".join(CATEGORIES)
    _state["translator"] = Translator()

    print("\n✅ Server ready — waiting for requests.\n")
    yield
    _state.clear()


app = FastAPI(title="Car Rental Review Inference", lifespan=lifespan)


class ReviewRequest(BaseModel):
    review: str


class ReviewResponse(BaseModel):
    review: str
    detected_language: str
    summary: str
    categories: str


class BatchReviewRequest(BaseModel):
    reviews: List[str]


def _run_inference(review: str) -> ReviewResponse:
    tokenizer = _state["tokenizer"]
    model = _state["model"]
    device = _state["device"]
    categories_str = _state["categories_str"]

    english_review, detected_language = _state["translator"].to_english(review)

    summary_prompt = f"Summarize the following car rental review.\n\n{english_review}\n\nSummary:"
    summary_inputs = tokenizer(summary_prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    summary_tokens = model.generate(
        input_ids=summary_inputs["input_ids"],
        max_new_tokens=60, do_sample=False,
        repetition_penalty=1.3, no_repeat_ngram_size=3,
    )[0]
    summary = tokenizer.decode(summary_tokens, skip_special_tokens=True)

    category_prompt = (
        f"Classify this car rental review into one or more of these categories: "
        f"{categories_str}.\n\nReview: {english_review}\n\nCategories:"
    )
    category_inputs = tokenizer(category_prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    category_tokens = model.generate(
        input_ids=category_inputs["input_ids"],
        max_new_tokens=40, do_sample=False,
        repetition_penalty=1.3, no_repeat_ngram_size=3,
    )[0]
    categories = tokenizer.decode(category_tokens, skip_special_tokens=True)

    return ReviewResponse(review=review, detected_language=detected_language, summary=summary, categories=categories)


def _run_batch_inference(reviews: List[str]) -> List[ReviewResponse]:
    tokenizer = _state["tokenizer"]
    model = _state["model"]
    device = _state["device"]
    categories_str = _state["categories_str"]

    translations = _state["translator"].batch_to_english(reviews)
    english_reviews = [t[0] for t in translations]
    detected_languages = [t[1] for t in translations]

    summary_prompts = [
        f"Summarize the following car rental review.\n\n{r}\n\nSummary:"
        for r in english_reviews
    ]
    category_prompts = [
        f"Classify this car rental review into one or more of these categories: "
        f"{categories_str}.\n\nReview: {r}\n\nCategories:"
        for r in english_reviews
    ]

    # One generate() call per task type instead of one per review
    summary_inputs = tokenizer(summary_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    summary_tokens = model.generate(
        input_ids=summary_inputs["input_ids"],
        attention_mask=summary_inputs["attention_mask"],
        max_new_tokens=60, do_sample=False,
        repetition_penalty=1.3, no_repeat_ngram_size=3,
    )
    summaries = tokenizer.batch_decode(summary_tokens, skip_special_tokens=True)

    category_inputs = tokenizer(category_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    category_tokens = model.generate(
        input_ids=category_inputs["input_ids"],
        attention_mask=category_inputs["attention_mask"],
        max_new_tokens=40, do_sample=False,
        repetition_penalty=1.3, no_repeat_ngram_size=3,
    )
    categories = tokenizer.batch_decode(category_tokens, skip_special_tokens=True)

    return [
        ReviewResponse(review=review, detected_language=lang, summary=summary, categories=cats)
        for review, lang, summary, cats in zip(reviews, detected_languages, summaries, categories)
    ]


@app.post("/infer", response_model=ReviewResponse)
def infer(req: ReviewRequest):
    return _run_inference(req.review)


@app.post("/infer/batch", response_model=List[ReviewResponse])
def infer_batch(req: BatchReviewRequest):
    return _run_batch_inference(req.reviews)


@app.get("/health")
def health():
    return {"status": "ok"}
