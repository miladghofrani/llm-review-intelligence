from config import CATEGORIES

# Keywords that signal each category. A review can match multiple categories.
CATEGORY_KEYWORDS = {
    "Vehicle Condition": [
        "dirty", "unclean", "filthy", "damage", "damaged", "scratch", "scratched",
        "dent", "dented", "smell", "smells", "smelly", "odor", "stain", "stained",
        "mechanical", "breakdown", "broke down", "engine", "tire", "flat tire",
        "broken", "malfunction", "defect", "defective", "worn", "old car",
        "maintenance", "warning light", "check engine", "interior", "exterior",
        "cigarette", "smoke", "pet hair", "debris", "rust",
    ],
    "Customer Service": [
        "staff", "agent", "employee", "representative", "rude", "polite",
        "helpful", "unhelpful", "attitude", "customer service", "support",
        "communication", "friendly", "unfriendly", "professional", "unprofessional",
        "manager", "counter", "ignored", "dismissive", "disrespectful",
        "wait time", "long wait", "no one helped",
    ],
    "Pricing & Billing": [
        "charge", "charged", "overcharge", "fee", "fees", "hidden fee",
        "price", "expensive", "cost", "billing", "invoice", "refund",
        "deposit", "credit card", "extra charge", "surcharge", "unauthorized",
        "fraud", "scam", "money", "payment", "receipt", "tax", "overpriced",
        "misleading price", "unexpected charge",
    ],
    "Insurance & Documents": [
        "insurance", "coverage", "liability", "contract", "agreement",
        "document", "fine print", "terms", "conditions", "policy",
        "waiver", "cdw", "collision", "damage waiver", "excess",
        "deductible", "claim", "accident report", "paperwork", "sign",
    ],
    "Pickup & Return": [
        "pickup", "pick up", "pick-up", "return", "drop off", "drop-off",
        "wait", "waiting", "queue", "long wait", "delay", "delayed",
        "location", "shuttle", "bus", "airport", "station", "ready",
        "not ready", "line", "check-in", "check-out", "key",
    ],
    "App & Booking": [
        "app", "application", "website", "online", "booking", "book",
        "reservation", "confirm", "confirmation", "email", "notification",
        "crash", "error", "bug", "glitch", "slow", "update",
        "interface", "digital", "portal", "system", "platform", "login",
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
