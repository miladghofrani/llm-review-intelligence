import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langdetect import detect, LangDetectException

TRANSLATION_MODELS = {
    "de": "Helsinki-NLP/opus-mt-de-en",
    "fr": "Helsinki-NLP/opus-mt-fr-en",
}


class Translator:
    def __init__(self, device: str):
        self._device = device
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._tokenizers = {}
        self._models = {}
        for lang, model_name in TRANSLATION_MODELS.items():
            self._tokenizers[lang] = AutoTokenizer.from_pretrained(model_name)
            self._models[lang] = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, dtype=dtype
            ).to(device)
            self._models[lang].eval()
        print("✅ Translation models loaded (de, fr → en)")

    def _detect(self, text: str) -> str:
        try:
            return detect(text)
        except LangDetectException:
            return "en"

    def _translate(self, lang: str, texts: list[str]) -> list[str]:
        tokenizer = self._tokenizers[lang]
        model = self._models[lang]
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self._device)
        with torch.no_grad():
            tokens = model.generate(**inputs, max_new_tokens=512)
        return tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def to_english(self, text: str) -> tuple[str, str]:
        """Returns (english_text, detected_language)."""
        lang = self._detect(text)
        if lang not in self._models:
            return text, lang
        return self._translate(lang, [text])[0], lang

    def batch_to_english(self, texts: list[str]) -> list[tuple[str, str]]:
        """Translate a batch, grouping by language for efficiency."""
        langs = [self._detect(t) for t in texts]

        groups: dict[str, list[int]] = {}
        for i, lang in enumerate(langs):
            groups.setdefault(lang, []).append(i)

        translated = list(texts)
        for lang, indices in groups.items():
            if lang not in self._models:
                continue
            results = self._translate(lang, [texts[i] for i in indices])
            for i, result in zip(indices, results):
                translated[i] = result

        return list(zip(translated, langs))
