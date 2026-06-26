from config import CATEGORIES

# Keywords that signal each category. A review can match multiple categories.
CATEGORY_KEYWORDS = {
    "Cleanliness": [
        "dirty", "unclean", "filthy", "clean", "spotless", "smell", "smells",
        "smelly", "odor", "stain", "stained", "cigarette", "smoke", "pet hair",
        "debris", "dusty", "greasy", "sticky", "trash", "garbage", "hygiene",
        "disgusting", "immaculate", "fresh", "washed",
    ],
    "Vehicle Condition": [
        "damage", "damaged", "scratch", "scratched", "dent", "dented", "rust",
        "mechanical", "breakdown", "broke down", "engine", "tire", "flat tire",
        "broken", "malfunction", "defect", "defective", "worn", "old car",
        "warning light", "check engine", "bald tires", "brakes", "noise",
        "accident", "recalled", "unsafe", "rattling", "wrong car", "model",
    ],
    "Pickup Experience": [
        "pickup", "pick up", "pick-up", "collection", "queue", "line", "wait",
        "waiting", "long wait", "delay", "delayed", "counter", "desk", "shuttle",
        "bus", "airport", "station", "ready", "not ready", "check-in", "key",
        "fast pickup", "smooth pickup", "slow", "crowded",
    ],
    "Return Experience": [
        "return", "drop off", "drop-off", "return process", "return desk",
        "damage inspection", "damage claim", "false damage", "dispute",
        "deposit refund", "refund", "charge after return", "check-out",
        "return location", "after hours", "key drop", "receipt",
    ],
    "Hidden Fees & Billing": [
        "hidden fee", "hidden charge", "extra charge", "unexpected charge",
        "overcharge", "surcharge", "unauthorized charge", "billing error",
        "invoice", "receipt", "deposit", "credit card", "fraud", "scam",
        "misleading price", "price difference", "overpriced", "expensive",
        "fuel charge", "toll charge", "administration fee",
    ],
    "Insurance & Upselling": [
        "insurance", "coverage", "liability", "waiver", "cdw", "collision",
        "damage waiver", "excess", "deductible", "upsell", "pressure",
        "pushed", "forced", "mandatory", "optional", "decline", "policy",
        "fine print", "terms", "claim", "accident report",
    ],
    "Staff & Communication": [
        "staff", "agent", "employee", "representative", "rude", "polite",
        "helpful", "unhelpful", "friendly", "unfriendly", "professional",
        "unprofessional", "attitude", "ignored", "dismissive", "disrespectful",
        "language", "english", "communication", "manager", "helpful staff",
        "great service", "poor service", "no one helped",
    ],
    "Booking & App": [
        "app", "application", "website", "online", "booking", "book",
        "reservation", "confirm", "confirmation", "email", "notification",
        "crash", "error", "bug", "glitch", "slow", "update", "interface",
        "portal", "login", "voucher", "change", "cancellation", "modify",
    ],
}


def label_review(text: str) -> list:
    """Returns all matching categories for a review text. Falls back to Customer Service."""
    text_lower = text.lower()
    matched = [
        category
        for category, keywords in CATEGORY_KEYWORDS.items()
        if any(keyword in text_lower for keyword in keywords)
    ]
    return matched if matched else ["Customer Service"]


def format_labels(labels: list) -> str:
    """Formats a list of category labels as a comma-separated string for model output."""
    return ", ".join(labels)
