from langdetect import detect, LangDetectException


class LanguageDetector:
    def detect(self, text: str) -> str:
        try:
            return detect(text)
        except LangDetectException:
            return "en"

    def batch_detect(self, texts: list[str]) -> list[str]:
        return [self.detect(t) for t in texts]
