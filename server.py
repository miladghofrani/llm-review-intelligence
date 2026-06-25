from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from config import ADAPTER_PATH, CATEGORIES
from device_utils import get_device
from model_loader import load_llm_model, load_tokenizer
from peft_trainer import load_saved_peft_model

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

    print("\n✅ Server ready — waiting for requests.\n")
    yield
    _state.clear()


app = FastAPI(title="Car Rental Review Inference", lifespan=lifespan)


class ReviewRequest(BaseModel):
    review: str


class ReviewResponse(BaseModel):
    summary: str
    categories: str


@app.post("/infer", response_model=ReviewResponse)
def infer(req: ReviewRequest):
    tokenizer = _state["tokenizer"]
    model = _state["model"]
    device = _state["device"]
    categories_str = _state["categories_str"]

    # Summarization
    summary_prompt = f"Summarize the following car rental review.\n\n{req.review}\n\nSummary:"
    summary_inputs = tokenizer(summary_prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    summary_tokens = model.generate(input_ids=summary_inputs["input_ids"], max_new_tokens=60, do_sample=False)[0]
    summary = tokenizer.decode(summary_tokens, skip_special_tokens=True)

    # Classification
    category_prompt = (
        f"Classify this car rental review into one or more of these categories: "
        f"{categories_str}.\n\nReview: {req.review}\n\nCategories:"
    )
    category_inputs = tokenizer(category_prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    category_tokens = model.generate(input_ids=category_inputs["input_ids"], max_new_tokens=40, do_sample=False)[0]
    categories = tokenizer.decode(category_tokens, skip_special_tokens=True)

    return ReviewResponse(summary=summary, categories=categories)


@app.get("/health")
def health():
    return {"status": "ok"}
