"""
Scholara NLP Processor
Uses multilingual sentence embeddings (paraphrase-multilingual-MiniLM-L12-v2)
for semantic intent detection. Works with Arabic, English, Arabizi, typos, and
mixed-language input. Runs fully offline — no external API required.
"""

import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class EgyptianStudentNLP:
    # Prototype sentences per intent (Arabic + English so the model is bilingual)
    INTENT_PROTOTYPES = {
        "predict_performance": [
            # Arabic / Colloquial Egyptian
            "انا درجتي 85 ف الاسايمنت وهل هنجح ولا لا",
            "جبت 60 في الميد تيرم تفتكر هعدي",
            "خايف اسقط عشان درجاتي وحشه",
            "درجتي 50 في الكويز عايز اعرف نتيجتي النهائيه",
            "نسبه نجاحي كام لو جبت 70",
            "هنجح ولا هشيل الماده دي",
            "انا ساقط ولا ناجح",
            "قولي هنجح ولا هسقط",
            "لو سمحت درجاتي واقعه هل هعدي",
            "انا جبت 90 هل كدا انا ممتاز",
            "هل درجتي كافية عشان أنجح",
            "هل هعدي الماده بالدرجات دي",
            # English equivalents
            "will I pass with this grade",
            "my score is 85 will I pass",
            "I got 60 in midterm will I succeed",
            "am I going to fail",
            "what are my chances of passing",
            "is my grade enough to pass",
            "I scored 75 do you think I will make it",
            "predict my result based on my grades",
        ],
        "get_advice": [
            # Arabic / Colloquial Egyptian
            "عايز حل عشان احسن درجاتي",
            "تنصحني بايه عشان اذاكر كويس",
            "عندي مشكله في المذاكره وعايز نصيحه",
            "ازاي اجيب درجات عاليه",
            "حاسس بضغط ومش عارف اركز اعمل ايه",
            "نصايح للمذاكره قبل الفاينل",
            "ياريت تقولي طريقه اذاكر بيها",
            "محتاج خطه للمذاكره",
            "عايز طريقه افهم بيها الشرح",
            "مش عارف اذاكر بليز ساعدني",
            "إزاي أرفع معدلي",
            "محتاج مساعدة في المذاكرة",
            "ايه أحسن طريقة للمراجعة",
            # English equivalents
            "how can I improve my grades",
            "give me study tips",
            "I need advice on how to study",
            "help me get better results",
            "I feel stressed how do I focus",
            "what is the best way to study",
            "I need a study plan",
            "tips for studying before exams",
            "how to boost my GPA",
            "I am struggling with my studies please help",
        ],
    }

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Loads the multilingual sentence-transformer model and pre-encodes
        all prototype sentences so similarity lookups are fast at runtime.
        """
        self._model = SentenceTransformer(model_name)

        # Pre-encode all prototypes
        self._intent_embeddings: dict[str, np.ndarray] = {}
        for intent, sentences in self.INTENT_PROTOTYPES.items():
            self._intent_embeddings[intent] = self._model.encode(
                sentences, convert_to_numpy=True
            )

    # ------------------------------------------------------------------
    # Public Methods
    # ------------------------------------------------------------------

    def analyze_input(self, user_text: str) -> dict:
        """
        Main entry point.
        Returns a dict with:
          - text            : original input
          - intent          : detected intent label
          - confidence      : cosine similarity score (0-1)
          - extracted_scores: list of integers found in the text
          - is_greeting     : True if input is a simple greeting
        """
        cleaned_text = re.sub(r'[^\w\s]', '', user_text).strip().lower()
        greetings_ar = ["اهلا", "اهلاً", "أهلاً", "سلام", "سلام عليكم", "مرحبا", "مرحباً", "صباح الخير", "مساء الخير", "يا هلا", "هلو"]
        greetings_en = ["hi", "hello", "hey", "greetings", "good morning", "good evening", "yo", "hello there"]

        is_greeting = cleaned_text in greetings_ar or cleaned_text in greetings_en or any(g in cleaned_text for g in ["أهلاً بك", "اهلا بك", "ازيك"])

        if is_greeting:
            return {
                "text": user_text,
                "intent": "get_advice",
                "confidence": 0.0,
                "extracted_scores": [],
                "is_greeting": True
            }

        embedding = self._model.encode([user_text], convert_to_numpy=True)

        best_intent = None
        best_score = -1.0

        for intent, proto_embeddings in self._intent_embeddings.items():
            sims = cosine_similarity(embedding, proto_embeddings)[0]
            max_sim = float(np.max(sims))
            if max_sim > best_score:
                best_score = max_sim
                best_intent = intent

        # Fallback: if confidence is too low, treat as generic "get_advice"
        if best_score < 0.25:
            best_intent = "get_advice"

        return {
            "text": user_text,
            "intent": best_intent,
            "confidence": round(best_score, 3),
            "extracted_scores": self._extract_numbers(user_text),
            "is_greeting": False
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_numbers(self, text: str) -> list[int]:
        """Extract all integers from text (handles Arabic-Indic digits too)."""
        # Convert Arabic-Indic digits → Western
        text = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
        return [int(n) for n in re.findall(r"\d+", text)]
