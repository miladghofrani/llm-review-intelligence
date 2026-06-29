import os
import deepl
from langdetect import detect, LangDetectException


class Translator:
    def __init__(self):
        api_key = os.environ.get("DEEPL_API_KEY", "")
        if not api_key:
            raise RuntimeError("DEEPL_API_KEY not set in environment")
        self._client = deepl.Translator(api_key)
        print("✅ DeepL translator ready")

    def _detect(self, text: str) -> str:
        try:
            return detect(text)
        except LangDetectException:
            return "en"

    def to_english(self, text: str) -> tuple[str, str]:
        """Returns (english_text, detected_language)."""
        lang = self._detect(text)
        if lang == "en":
            return text, "en"
        result = self._client.translate_text(text, target_lang="EN-US")
        return result.text, result.detected_source_lang.lower()

    def batch_to_english(self, texts: list[str]) -> list[tuple[str, str]]:
        """Batch translate, skipping English texts to save API quota."""
        langs = [self._detect(t) for t in texts]

        non_en_indices = [i for i, l in enumerate(langs) if l != "en"]
        non_en_texts = [texts[i] for i in non_en_indices]

        translated = list(texts)
        result_langs = list(langs)

        if non_en_texts:
            results = self._client.translate_text(non_en_texts, target_lang="EN-US")
            for i, result in zip(non_en_indices, results):
                translated[i] = result.text
                result_langs[i] = result.detected_source_lang.lower()

        return list(zip(translated, result_langs))
