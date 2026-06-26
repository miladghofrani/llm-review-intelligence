from transformers import pipeline as hf_pipeline
from langdetect import detect, LangDetectException

TRANSLATION_MODELS = {
    "de": "Helsinki-NLP/opus-mt-de-en",
    "fr": "Helsinki-NLP/opus-mt-fr-en",
}


class Translator:
    def __init__(self, device: str):
        pipeline_device = 0 if device == "cuda" else -1
        self._pipelines = {
            lang: hf_pipeline("translation", model=model, device=pipeline_device)
            for lang, model in TRANSLATION_MODELS.items()
        }
        print("✅ Translation models loaded (de, fr → en)")

    def _detect(self, text: str) -> str:
        try:
            return detect(text)
        except LangDetectException:
            return "en"

    def to_english(self, text: str) -> tuple[str, str]:
        """Returns (english_text, detected_language)."""
        lang = self._detect(text)
        if lang not in self._pipelines:
            return text, lang
        result = self._pipelines[lang](text, max_length=512)
        return result[0]["translation_text"], lang

    def batch_to_english(self, texts: list[str]) -> list[tuple[str, str]]:
        """Translate a batch, grouping by language for efficiency."""
        langs = [self._detect(t) for t in texts]

        groups: dict[str, list[int]] = {}
        for i, lang in enumerate(langs):
            groups.setdefault(lang, []).append(i)

        translated = list(texts)
        for lang, indices in groups.items():
            if lang not in self._pipelines:
                continue
            batch = [texts[i] for i in indices]
            results = self._pipelines[lang](batch, max_length=512, batch_size=len(batch))
            for i, r in zip(indices, results):
                translated[i] = r["translation_text"]

        return list(zip(translated, langs))
