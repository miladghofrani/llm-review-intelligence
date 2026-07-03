from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

import torch

from config import ADAPTER_PATH, CATEGORIES
from .device_utils import get_device
from .model_loader import load_llm_model, load_tokenizer, load_mbart_model, load_mbart_tokenizer
from .peft_trainer import load_saved_peft_model
from .translator import Translator

_state: dict = {}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_categories(raw: str) -> List[str]:
    """Split model output and keep only recognised category names."""
    valid = set(CATEGORIES)
    return [c.strip() for c in raw.split(",") if c.strip() in valid]


def _parse_sentiment(raw: str) -> str:
    t = raw.strip().lower()
    if "positive" in t:
        return "positive"
    if "negative" in t:
        return "negative"
    return "mixed"


def _category_flags(categories: List[str]) -> dict:
    return {
        "has_damage_claim": "Vehicle Condition" in categories,
        "has_hidden_fees":  "Hidden Fees & Billing" in categories,
        "has_upselling":    "Insurance & Upselling" in categories,
    }


# ── Models ─────────────────────────────────────────────────────────────────────

class ReviewRequest(BaseModel):
    review: str
    # Floyt metadata — passed through to the Elasticsearch output as-is.
    # All fields are optional so the endpoint also works without Floyt context.
    database_id: Optional[int] = None
    provider: Optional[str] = None       # intermediary, e.g. "BSPAuto"
    provider_id: Optional[str] = None
    renter: Optional[str] = None         # on-site company, e.g. "Dollar"
    location: Optional[str] = None       # pickup city
    departure: Optional[str] = None      # airport / station code
    country_code: Optional[str] = None
    aggregate_rating: Optional[float] = None
    renter_rating: Optional[float] = None


class ElasticsearchDoc(BaseModel):
    """
    Ready-to-index document that merges the original Floyt metadata with
    AI-generated enrichments. Store this document to enable filtering and
    aggregations across providers, sentiment, issue types, and locations.
    """
    # ── Floyt passthrough ──
    database_id: Optional[int]
    provider: Optional[str]
    provider_id: Optional[str]
    renter: Optional[str]
    location: Optional[str]
    departure: Optional[str]
    country_code: Optional[str]
    aggregate_rating: Optional[float]
    renter_rating: Optional[float]
    # ── AI-generated ──
    language: str                   # original language code (de, fr, en, …)
    sentiment: str                  # positive | negative | mixed
    primary_category: str           # top category for quick filtering
    categories: List[str]
    has_damage_claim: bool
    has_hidden_fees: bool
    has_upselling: bool


class ReviewAnalysis(BaseModel):
    original_review: str
    detected_language: str
    english_translation: Optional[str]  # null when review was already in English
    summary: str
    sentiment: str
    categories: List[str]
    elasticsearch: ElasticsearchDoc


class BatchReviewRequest(BaseModel):
    reviews: List[ReviewRequest]


# ── App ────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    device = get_device()

    # flan-t5-large — categories + sentiment
    tokenizer  = load_tokenizer()
    base_model = load_llm_model(device)
    model      = load_saved_peft_model(device, base_model, ADAPTER_PATH)
    model.eval()

    # mBART — summarization
    mbart_tokenizer = load_mbart_tokenizer()
    mbart_model     = load_mbart_model(device)

    _state["device"]          = device
    _state["tokenizer"]       = tokenizer
    _state["model"]           = model
    _state["categories_str"]  = ", ".join(CATEGORIES)
    _state["translator"]      = Translator()
    _state["mbart_tokenizer"] = mbart_tokenizer
    _state["mbart_model"]     = mbart_model

    print("\n✅ Server ready — waiting for requests.\n")
    yield
    _state.clear()


app = FastAPI(title="Car Rental Review Inference", lifespan=lifespan)


# ── Inference ──────────────────────────────────────────────────────────────────

def _build_analysis(req: ReviewRequest, english_review: str, detected_language: str,
                    summary: str, categories: List[str], sentiment: str) -> ReviewAnalysis:
    flags = _category_flags(categories)
    return ReviewAnalysis(
        original_review=req.review,
        detected_language=detected_language,
        english_translation=english_review if detected_language != "en" else None,
        summary=summary,
        sentiment=sentiment,
        categories=categories,
        elasticsearch=ElasticsearchDoc(
            database_id=req.database_id,
            provider=req.provider,
            provider_id=req.provider_id,
            renter=req.renter,
            location=req.location,
            departure=req.departure,
            country_code=req.country_code,
            aggregate_rating=req.aggregate_rating,
            renter_rating=req.renter_rating,
            language=detected_language,
            sentiment=sentiment,
            primary_category=categories[0] if categories else "Staff & Communication",
            categories=categories,
            **flags,
        ),
    )


def _mbart_summarize(review: str) -> str:
    tokenizer = _state["mbart_tokenizer"]
    model     = _state["mbart_model"]
    device    = _state["device"]
    tokenizer.src_lang = "de_DE"
    inputs = tokenizer(review, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],
            max_new_tokens=50,
            num_beams=4,
            no_repeat_ngram_size=3,
            length_penalty=0.6,
        )
    return tokenizer.decode(tokens[0], skip_special_tokens=True)


def _mbart_summarize_batch(reviews: List[str]) -> List[str]:
    tokenizer = _state["mbart_tokenizer"]
    model     = _state["mbart_model"]
    device    = _state["device"]
    tokenizer.src_lang = "de_DE"
    inputs = tokenizer(
        reviews, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)
    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],
            max_new_tokens=80,
        )
    return tokenizer.batch_decode(tokens, skip_special_tokens=True)


def _analyse(req: ReviewRequest) -> ReviewAnalysis:
    tokenizer      = _state["tokenizer"]
    model          = _state["model"]
    device         = _state["device"]
    categories_str = _state["categories_str"]

    english_review, detected_language = _state["translator"].to_english(req.review)

    def _generate(prompt: str, max_new_tokens: int, **kwargs) -> str:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        tokens = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            **kwargs,
        )[0]
        return tokenizer.decode(tokens, skip_special_tokens=True)

    summary    = _mbart_summarize(req.review)
    categories = _parse_categories(_generate(
        f"Classify this car rental review into one or more of these categories: "
        f"{categories_str}.\n\nReview: {req.review}\n\nCategories:",
        max_new_tokens=40, repetition_penalty=1.3, no_repeat_ngram_size=3,
    ))
    sentiment  = _parse_sentiment(_generate(
        f"What is the overall sentiment of this car rental review? "
        f"Answer with exactly one word: positive, negative, or mixed.\n\n"
        f"Review: {req.review}\n\nSentiment:",
        max_new_tokens=5,
    ))

    return _build_analysis(req, english_review, detected_language, summary, categories, sentiment)


def _analyse_batch(requests: List[ReviewRequest]) -> List[ReviewAnalysis]:
    tokenizer      = _state["tokenizer"]
    model          = _state["model"]
    device         = _state["device"]
    categories_str = _state["categories_str"]

    translations       = _state["translator"].batch_to_english([r.review for r in requests])
    english_reviews    = [t[0] for t in translations]
    detected_languages = [t[1] for t in translations]
    original_reviews   = [r.review for r in requests]

    def _batch_generate(prompts: List[str], max_new_tokens: int, **kwargs) -> List[str]:
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(device)
        tokens = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            **kwargs,
        )
        return tokenizer.batch_decode(tokens, skip_special_tokens=True)

    summaries      = _mbart_summarize_batch(original_reviews)
    categories_raw = _batch_generate(
        [
            f"Classify this car rental review into one or more of these categories: "
            f"{categories_str}.\n\nReview: {r}\n\nCategories:"
            for r in original_reviews
        ],
        max_new_tokens=40, repetition_penalty=1.3, no_repeat_ngram_size=3,
    )
    sentiments_raw = _batch_generate(
        [
            f"What is the overall sentiment of this car rental review? "
            f"Answer with exactly one word: positive, negative, or mixed.\n\n"
            f"Review: {r}\n\nSentiment:"
            for r in original_reviews
        ],
        max_new_tokens=5,
    )

    return [
        _build_analysis(
            req, eng, lang,
            summary,
            _parse_categories(cats_str),
            _parse_sentiment(sent_raw),
        )
        for req, eng, lang, summary, cats_str, sent_raw in zip(
            requests, english_reviews, detected_languages,
            summaries, categories_raw, sentiments_raw,
        )
    ]


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.post("/infer", response_model=ReviewAnalysis)
def infer(req: ReviewRequest):
    return _analyse(req)


@app.post("/infer/batch", response_model=List[ReviewAnalysis])
def infer_batch(req: BatchReviewRequest):
    return _analyse_batch(req.reviews)


@app.get("/health")
def health():
    return {"status": "ok"}
