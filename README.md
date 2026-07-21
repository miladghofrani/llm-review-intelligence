# Car Rental Review AI

A PEFT/LoRA fine-tuned `flan-t5-large` model that analyses car rental customer reviews — classifying the issue type, detecting sentiment, and detecting the original language (German, French, English) — with structured output ready to index into Elasticsearch.

Built for the Customer Experience team at [Billiger Mietwagen](https://www.billiger-mietwagen.de).

---

## What it does

Send a review (in any language) and get back:

```json
{
  "database_id": 489242,
  "language": "de",
  "sentiment": "negative",
  "primary_category": "Pickup Experience",
  "categories": ["Pickup Experience", "Insurance & Upselling", "Return Experience"],
  "has_damage_claim": false,
  "has_hidden_fees": false,
  "has_upselling": true
}
```

**Categories:** Cleanliness · Vehicle Condition · Pickup Experience · Return Experience · Hidden Fees & Billing · Insurance & Upselling · Staff & Communication · Booking & App

**Languages:** German, French, English (auto-detected via `langdetect`; the model classifies the review directly in its original language)

The response only contains `database_id` (to identify which Elasticsearch document to merge the enrichment into) plus what the model actually generated — the original review text, provider/renter/location, and ratings aren't echoed back since the caller already has them.

---

## Project structure

```
├── app/                        # FastAPI inference server (what Docker runs)
│   ├── server.py               # endpoints, request/response models
│   ├── model_loader.py         # loads flan-t5-large base model
│   ├── peft_trainer.py         # LoRA adapter injection and loading
│   ├── language_detector.py    # langdetect wrapper for original review language
│   └── device_utils.py         # CPU / CUDA detection
│
├── data_processing/            # training data pipeline
│   ├── loader.py               # loads dataset from HF Hub or local JSONL
│   ├── preprocessor.py         # tokenisation with -100 label padding
│   └── labeler.py              # keyword-based category fallback
│
├── scripts/                    # one-off tools, not imported by the server
│   ├── train.py                # local training entrypoint
│   ├── generate_dataset.py     # generate synthetic reviews via Groq API
│   ├── label_reviews.py        # label real Floyt reviews with sentiment + categories via Claude
│   └── push_dataset.py         # push JSONL dataset to HuggingFace Hub
│
├── floyt/                      # Elasticsearch / Floyt integration helpers
│   ├── reviews.json            # sample export from the customer_reviews index
│   ├── convert.py              # converts Floyt JSON → batch_request.json
│   └── batch_request.json      # ready-to-use curl payload
│
├── notebooks/
│   └── classification.ipynb    # Kaggle training notebook (flan-t5-large, 3 epochs)
│
├── config.py                   # MODEL_NAME, ADAPTER_PATH, CATEGORIES, training params
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── .env.example
```

---

## Quick start

**1. Set up environment variables**

```bash
cp .env.example .env
# Fill in HF_TOKEN, GROQ_API_KEY
```

**2. Build and start**

```bash
make setup    # builds the Docker image
make start    # starts the container and waits until healthy
```

The server starts on **http://localhost:8742**.

**3. Test with a single review**

```bash
curl -X POST http://localhost:8742/infer \
  -H "Content-Type: application/json" \
  -d '{"review": "Das Auto war eine Klapperkiste und der Mitarbeiter sehr unhöflich."}'
```

**4. Test with Floyt data**

```bash
python3 floyt/convert.py    # generates floyt/batch_request.json from reviews.json

curl -X POST http://localhost:8742/infer/batch \
  -H "Content-Type: application/json" \
  -d @floyt/batch_request.json
```

---

## API reference

### `POST /infer`

Analyse a single review. `database_id` and the rating fields are optional — `database_id` is echoed back in the response as the merge key, while the ratings feed the averages/NPS computed in `/infer/aggregate` (not returned here).

```json
{
  "review": "string (required)",
  "database_id": 489244,
  "aggregate_rating": 2.5,
  "renter_rating": 2.33,
  "car_condition_rating": 4,
  "processing_speed_rating": 3,
  "provider_care_rating": 5,
  "service_level_rating": 4,
  "recommendation_rating": 8
}
```

### `POST /infer/batch`

Same fields per review, wrapped in a list:

```json
{
  "reviews": [ { "review": "...", "database_id": 489244, ... }, ... ]
}
```

Batching runs 2 `model.generate()` calls (categories, sentiment) over all reviews at once — significantly faster than calling `/infer` in a loop.

### `GET /health`

Returns `{"status": "ok"}` when the model is loaded and ready.

---

## Environment variables

See `.env.example` for the full list with descriptions. Required keys:

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | Pull the LoRA adapter from HuggingFace Hub on startup |
| `GROQ_API_KEY` | Generate synthetic training data (only needed for `generate_dataset.py`) |

---

## Model and training

**Model:** `google/flan-t5-large` (780M params) fine-tuned with LoRA rank 16 on 10,000 synthetic car rental reviews — 1.4% of parameters are trainable.

**Adapter:** [`miladghofrani/car-rental-peft-adapter-large`](https://huggingface.co/miladghofrani/car-rental-peft-adapter-large) on HuggingFace Hub. Loaded automatically on container startup via `HF_TOKEN`.

**Dataset:** [`miladghofrani/car-rental-reviews`](https://huggingface.co/datasets/miladghofrani/car-rental-reviews) — 10,000 synthetic reviews (40% negative, 40% positive, 20% mixed) generated with Llama 3.3 70B via Groq API, covering all 8 categories across European rental locations.

### Retrain the model

Training runs on Kaggle (free T4 GPU). Open `notebooks/classification.ipynb`, set your secrets (`HF_TOKEN`), and run all cells. The notebook trains for 3 epochs and pushes the updated adapter to HuggingFace Hub.

To generate additional training data locally:

```bash
python3 scripts/generate_dataset.py    # resumes from current count, targets 10,000 total
python3 scripts/push_dataset.py        # pushes JSONL to HuggingFace Hub
```

**When to retrain:** Only when adding or changing categories, or when you have accumulated a significant number of real labelled reviews to add to the training set. The model does not need periodic retraining for the same task.

---

## AWS deployment

### Recommended stack

| Component | Service | Notes |
|---|---|---|
| Inference server | ECS + Fargate | Managed containers, no servers to maintain |
| Container registry | ECR | Stores your Docker image |
| Data & model backup | S3 | Training JSONL, adapter weights snapshot |
| Retraining (when needed) | SageMaker Training Jobs | Pay-per-use GPU, shuts down after training |

### Why not Lambda or SageMaker Endpoints?

**Lambda** — the model takes 30–60 seconds to load. Cold starts make it unusable for ML inference.

**SageMaker Endpoints** — purpose-built for high-traffic ML APIs, priced accordingly. At this scale (CE team usage), ECS + Fargate costs 2–3× less for the same compute.

**Bedrock** — provides hosted foundation models via API. Not needed here: you have a fine-tuned model, you control it, and EU customer review data stays within your infrastructure.

### Step-by-step

**1. Push image to ECR**

```bash
aws ecr create-repository --repository-name car-rental-review-ai

aws ecr get-login-password --region eu-central-1 \
  | docker login --username AWS --password-stdin \
    <account-id>.dkr.ecr.eu-central-1.amazonaws.com

docker build -t car-rental-review-ai .
docker tag car-rental-review-ai:latest \
  <account-id>.dkr.ecr.eu-central-1.amazonaws.com/car-rental-review-ai:latest
docker push \
  <account-id>.dkr.ecr.eu-central-1.amazonaws.com/car-rental-review-ai:latest
```

**2. Create ECS cluster and task definition**

- Launch type: **Fargate**
- CPU: `1024` (1 vCPU), Memory: `4096` MB minimum — `8192` MB recommended for `flan-t5-large`
- Container port: `8742`
- Environment variables: `HF_TOKEN` — store in **AWS Secrets Manager** and reference it in the task definition (never hardcode)

**3. Create a service**

- Attach to an Application Load Balancer on port 443 (HTTPS)
- Health check path: `/health`
- Desired count: `1` (scale up if needed)

**4. Retraining with SageMaker (when needed)**

SageMaker Training Jobs spin up a GPU instance, run training, push the updated adapter to S3 or HuggingFace Hub, then shut down — you pay only for the training duration (~3–4 hours at ~€0.60/hour for `ml.g4dn.xlarge`).

After the new adapter is pushed to HuggingFace Hub, restart the ECS service to pick it up:

```bash
aws ecs update-service \
  --cluster car-rental-cluster \
  --service car-rental-review-ai \
  --force-new-deployment
```

### Instance sizing

`flan-t5-large` runs on CPU. The model needs ~3 GB RAM; allow overhead for the FastAPI process and tokenisation.

| Fargate config | RAM | Suitable for |
|---|---|---|
| 1 vCPU / 4 GB | Tight | Low traffic, single request at a time |
| 1 vCPU / 8 GB | Comfortable | Recommended starting point |
| 2 vCPU / 8 GB | Headroom | Concurrent batch requests |

---

## Local development

```bash
make logs       # tail container logs
make console    # bash shell inside the running container
make stop       # stop the container
make clean      # remove container, image, and volumes (including model cache)
```

After any code change, restart the container to reload:

```bash
make stop && make start
```

The `hf-cache` Docker volume persists the downloaded model weights between restarts — the model only downloads on the first run.
