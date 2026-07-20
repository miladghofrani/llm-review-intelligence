from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from config import ADAPTER_PATH, CATEGORIES

from .device_utils import get_device
from .language_detector import LanguageDetector
from .model_loader import load_llm_model, load_tokenizer
from .peft_trainer import load_saved_peft_model

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
    database_id: Optional[int] = None    # identifies the ES document to update with the enrichment
    provider: Optional[str] = None       # intermediary, e.g. "BSPAuto"
    renter: Optional[str] = None         # on-site company, e.g. "Dollar"
    location: Optional[str] = None       # pickup city
    aggregate_rating: Optional[float] = None
    renter_rating: Optional[float] = None
    # Elasticsearch rating fields — customer-given scores per dimension.
    car_condition_rating: Optional[float] = None
    processing_speed_rating: Optional[float] = None
    provider_care_rating: Optional[float] = None
    service_level_rating: Optional[float] = None
    recommendation_rating: Optional[float] = None   # 0-10 NPS-style score


class ElasticsearchDoc(BaseModel):
    """
    Ready-to-index document that merges the original Floyt metadata with
    AI-generated enrichments. Store this document to enable filtering and
    aggregations across providers, sentiment, issue types, and locations.
    """
    # ── Floyt passthrough ──
    database_id: Optional[int]
    provider: Optional[str]
    renter: Optional[str]
    location: Optional[str]
    aggregate_rating: Optional[float]
    renter_rating: Optional[float]
    car_condition_rating: Optional[float]
    processing_speed_rating: Optional[float]
    provider_care_rating: Optional[float]
    service_level_rating: Optional[float]
    recommendation_rating: Optional[float]
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
    sentiment: str
    categories: List[str]
    elasticsearch: ElasticsearchDoc


class BatchReviewRequest(BaseModel):
    reviews: List[ReviewRequest]


class AggregateRequest(BaseModel):
    reviews: List[ReviewRequest]
    provider: Optional[str] = None
    location: Optional[str] = None
    renter: Optional[str] = None


class AggregateResponse(BaseModel):
    provider: Optional[str]
    location: Optional[str]
    renter: Optional[str]
    total_reviews: int
    sentiment_distribution: dict   # positive/negative/mixed counts + percentages
    top_categories: List[str]      # top 3 most mentioned categories
    flags: dict                    # upselling / hidden_fees / damage_claims counts
    ratings: dict                  # average of each customer-given rating dimension
    nps: dict                      # promoter/passive/detractor counts + NPS score
    aggregate_summary: str         # Amazon-style narrative paragraph


# ── App ────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    device = get_device()

    # flan-t5-large — categories + sentiment
    tokenizer  = load_tokenizer()
    base_model = load_llm_model(device)
    model      = load_saved_peft_model(device, base_model, ADAPTER_PATH)
    model.eval()

    _state["device"]            = device
    _state["tokenizer"]         = tokenizer
    _state["model"]             = model
    _state["categories_str"]    = ", ".join(CATEGORIES)
    _state["language_detector"] = LanguageDetector()

    print("\n✅ Server ready — waiting for requests.\n")
    yield
    _state.clear()


app = FastAPI(title="Car Rental Review Inference", lifespan=lifespan)


# ── Inference ──────────────────────────────────────────────────────────────────

def _build_analysis(req: ReviewRequest, detected_language: str,
                    categories: List[str], sentiment: str) -> ReviewAnalysis:
    flags = _category_flags(categories)
    return ReviewAnalysis(
        original_review=req.review,
        detected_language=detected_language,
        sentiment=sentiment,
        categories=categories,
        elasticsearch=ElasticsearchDoc(
            database_id=req.database_id,
            provider=req.provider,
            renter=req.renter,
            location=req.location,
            aggregate_rating=req.aggregate_rating,
            renter_rating=req.renter_rating,
            car_condition_rating=req.car_condition_rating,
            processing_speed_rating=req.processing_speed_rating,
            provider_care_rating=req.provider_care_rating,
            service_level_rating=req.service_level_rating,
            recommendation_rating=req.recommendation_rating,
            language=detected_language,
            sentiment=sentiment,
            primary_category=categories[0] if categories else "Staff & Communication",
            categories=categories,
            **flags,
        ),
    )


def _analyse(req: ReviewRequest) -> ReviewAnalysis:
    tokenizer      = _state["tokenizer"]
    model          = _state["model"]
    device         = _state["device"]
    categories_str = _state["categories_str"]

    detected_language = _state["language_detector"].detect(req.review)

    def _generate(prompt: str, max_new_tokens: int, **kwargs) -> str:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        tokens = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            **kwargs,
        )[0]
        return tokenizer.decode(tokens, skip_special_tokens=True)

    categories = _parse_categories(_generate(
        f"Classify this car rental review into 1-3 of these categories: {categories_str}.\n"
        f"Use 'Insurance & Upselling' if insurance or extra products were pushed or required.\n"
        f"Use 'Hidden Fees & Billing' if unexpected charges or fees are mentioned.\n\n"
        f"Review: {req.review}\n\nCategories:",
        max_new_tokens=40, repetition_penalty=1.3, no_repeat_ngram_size=3,
    ))
    sentiment  = _parse_sentiment(_generate(
        f"What is the overall sentiment of this car rental review? "
        f"Answer with exactly one word: positive, negative, or mixed.\n\n"
        f"Review: {req.review}\n\nSentiment:",
        max_new_tokens=5,
    ))

    return _build_analysis(req, detected_language, categories, sentiment)


def _analyse_batch(requests: List[ReviewRequest]) -> List[ReviewAnalysis]:
    tokenizer      = _state["tokenizer"]
    model          = _state["model"]
    device         = _state["device"]
    categories_str = _state["categories_str"]

    original_reviews   = [r.review for r in requests]
    detected_languages = _state["language_detector"].batch_detect(original_reviews)

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

    categories_raw = _batch_generate(
        [
            f"Classify this car rental review into 1-3 of these categories: {categories_str}.\n"
            f"Use 'Insurance & Upselling' if insurance or extra products were pushed or required.\n"
            f"Use 'Hidden Fees & Billing' if unexpected charges or fees are mentioned.\n\n"
            f"Review: {r}\n\nCategories:"
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
            req, lang,
            _parse_categories(cats_str),
            _parse_sentiment(sent_raw),
        )
        for req, lang, cats_str, sent_raw in zip(
            requests, detected_languages,
            categories_raw, sentiments_raw,
        )
    ]


# ── Aggregate ──────────────────────────────────────────────────────────────────

_CATEGORY_NATURAL = {
    "Staff & Communication":  "staff and communication",
    "Vehicle Condition":      "vehicle condition",
    "Pickup Experience":      "the pickup process",
    "Return Experience":      "the return process",
    "Hidden Fees & Billing":  "unexpected charges",
    "Insurance & Upselling":  "insurance upselling",
    "Cleanliness":            "vehicle cleanliness",
    "Booking & App":          "the booking experience",
}


RATING_FIELDS = [
    "aggregate_rating",
    "renter_rating",
    "car_condition_rating",
    "processing_speed_rating",
    "provider_care_rating",
    "service_level_rating",
    "recommendation_rating",
]


def _avg_ratings(results: List[ReviewAnalysis]) -> dict:
    averages = {}
    for field in RATING_FIELDS:
        values = [v for r in results if (v := getattr(r.elasticsearch, field)) is not None]
        averages[field] = round(sum(values) / len(values), 2) if values else None
    return averages


def _nps(results: List[ReviewAnalysis]) -> dict:
    scores = [r.elasticsearch.recommendation_rating for r in results
              if r.elasticsearch.recommendation_rating is not None]
    if not scores:
        return {"promoters": 0, "passives": 0, "detractors": 0, "score": None}

    promoters  = sum(1 for s in scores if s >= 9)
    detractors = sum(1 for s in scores if s <= 6)
    passives   = len(scores) - promoters - detractors

    return {
        "promoters":  promoters,
        "passives":   passives,
        "detractors": detractors,
        "score": round((promoters - detractors) / len(scores) * 100),
    }


def _build_aggregate_narrative(
    total: int, provider, location, renter,
    pos: int, neg: int, mixed: int,
    praised_cats: List[str], complaint_cats: List[str],
    upselling_count: int, fees_count: int, damage_count: int,
) -> str:
    parts = []
    if renter:
        parts.append(renter)
    elif provider:
        parts.append(provider)
    if location:
        parts.append(location)
    context = " at ".join(parts) if parts else "this rental location"

    pos_pct = round(100 * pos / total) if total else 0
    neg_pct = round(100 * neg / total) if total else 0

    sentences = []

    # Positive part
    if praised_cats and pos_pct > 15:
        phrases = [_CATEGORY_NATURAL.get(c, c.lower()) for c in praised_cats[:2]]
        praised_str = f"{phrases[0]} and {phrases[1]}" if len(phrases) > 1 else phrases[0]
        sentences.append(
            f"Customers appreciate the {praised_str} at {context}."
        )
    else:
        sentences.append(f"Customers have shared their experiences with {context}.")

    # Negative / flag part
    issue_phrases = []
    if complaint_cats and neg_pct > 15:
        phrases = [_CATEGORY_NATURAL.get(c, c.lower()) for c in complaint_cats[:2]]
        issue_phrases.extend(phrases)

    flag_phrases = []
    if upselling_count:
        flag_phrases.append("insurance upselling")
    if fees_count:
        flag_phrases.append("unexpected charges")
    if damage_count:
        flag_phrases.append("disputed damage claims")

    # Merge complaint categories and flags, deduplicate
    all_issues = list(dict.fromkeys(issue_phrases + flag_phrases))[:3]

    if all_issues:
        issues_str = ", ".join(all_issues[:-1]) + (" and " if len(all_issues) > 1 else "") + all_issues[-1]
        sentences.append(
            f"However, some reviewers report issues with {issues_str}."
        )

    return " ".join(sentences)


def _aggregate(req: AggregateRequest) -> AggregateResponse:
    from collections import Counter

    results = _analyse_batch(req.reviews)
    total = len(results)

    sentiment_counts = {"positive": 0, "negative": 0, "mixed": 0}
    all_cats: Counter = Counter()
    positive_cats: Counter = Counter()
    negative_cats: Counter = Counter()

    for r in results:
        sentiment_counts[r.sentiment] = sentiment_counts.get(r.sentiment, 0) + 1
        for cat in r.categories:
            all_cats[cat] += 1
            if r.sentiment == "positive":
                positive_cats[cat] += 1
            elif r.sentiment == "negative":
                negative_cats[cat] += 1

    top_categories = [c for c, _ in all_cats.most_common(3)]
    praised    = [c for c, _ in positive_cats.most_common(2)]
    complained = [c for c, _ in negative_cats.most_common(2)]

    upselling_count  = sum(1 for r in results if r.elasticsearch.has_upselling)
    hidden_fees_count = sum(1 for r in results if r.elasticsearch.has_hidden_fees)
    damage_count     = sum(1 for r in results if r.elasticsearch.has_damage_claim)

    pos   = sentiment_counts["positive"]
    neg   = sentiment_counts["negative"]
    mixed = sentiment_counts["mixed"]

    ratings = _avg_ratings(results)
    nps     = _nps(results)

    narrative = _build_aggregate_narrative(
        total=total, provider=req.provider, location=req.location, renter=req.renter,
        pos=pos, neg=neg, mixed=mixed,
        praised_cats=praised, complaint_cats=complained,
        upselling_count=upselling_count, fees_count=hidden_fees_count, damage_count=damage_count,
    )

    return AggregateResponse(
        provider=req.provider,
        location=req.location,
        renter=req.renter,
        total_reviews=total,
        sentiment_distribution={
            "positive":     pos,
            "negative":     neg,
            "mixed":        mixed,
            "positive_pct": round(100 * pos / total) if total else 0,
            "negative_pct": round(100 * neg / total) if total else 0,
            "mixed_pct":    round(100 * mixed / total) if total else 0,
        },
        top_categories=top_categories,
        flags={
            "upselling_count":   upselling_count,
            "hidden_fees_count": hidden_fees_count,
            "damage_claims_count": damage_count,
        },
        ratings=ratings,
        nps=nps,
        aggregate_summary=narrative,
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.post("/infer", response_model=ReviewAnalysis)
def infer(req: ReviewRequest):
    return _analyse(req)


@app.post("/infer/batch", response_model=List[ReviewAnalysis])
def infer_batch(req: BatchReviewRequest):
    return _analyse_batch(req.reviews)


@app.post("/infer/aggregate", response_model=AggregateResponse)
def infer_aggregate(req: AggregateRequest):
    return _aggregate(req)


@app.get("/health")
def health():
    return {"status": "ok"}
