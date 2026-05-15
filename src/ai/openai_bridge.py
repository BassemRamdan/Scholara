"""
OpenAI Bridge — integrates GPT-4o-mini into the EduGuide pipeline.

Two roles:
  1. PRE-PROCESSOR  — normalizes messy Arabic / Arabizi / mixed-language text
                      into clean Modern Standard Arabic so the local
                      sentence-transformer model works at its best.

  2. FALLBACK       — when the local model's confidence is below threshold,
                      GPT-4o-mini answers the student directly (in Egyptian Arabic).
"""

from __future__ import annotations
import os
from openai import OpenAI


# ── System prompt shared by all calls ─────────────────────────────────────
_SYSTEM_BASE = (
    "أنت EduGuide، مساعد ذكي للطلاب المصريين. "
    "بتتكلم دايماً بالعربي المصري العامية. "
    "ردودك قصيرة ومفيدة وودية."
)

_NORMALIZE_SYSTEM = (
    "أنت مساعد لتنظيف النصوص. "
    "مهمتك: حوّل الجملة دي — سواء كانت عربية عامية أو إنجليزي أو مزيج — "
    "لعربي فصيح واضح يصف نفس المعنى. "
    "أرجع الجملة المنظفة فقط، بدون أي شرح أو تعليق."
)

_EXTRACT_SYSTEM = (
    "You are a strict data extractor. Extract the student's scores ONLY IF the user explicitly mentions the subject name.\n"
    "Allowed subject keys: Midterm_Score, Final_Score, Assignments_Avg, Quizzes_Avg, Participation_Score, Projects_Score, Attendance (%)\n"
    "Rule 1: Ignore max possible marks (e.g., 'out of 30' or 'من 50'). Extract ONLY the score the student achieved.\n"
    "Rule 2: If the user provides a bare number (e.g., '20', '50') WITHOUT explicitly naming the subject, DO NOT GUESS. You MUST return {}.\n"
    "Return JSON only (e.g., {\"Midterm_Score\": 20}). If no clear subject is found, return {}."
)


class OpenAIBridge:
    """
    Wraps OpenAI calls with two jobs:
      - normalize(text)         → clean text for local NLP
      - generate_response(...)  → fallback Egyptian Arabic answer
    """

    # Confidence score below which we fall back to OpenAI
    FALLBACK_THRESHOLD = 0.45

    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError(
                "OpenAI API key not found. "
                "Set OPENAI_API_KEY in .streamlit/secrets.toml or as an env variable."
            )
        self._client = OpenAI(api_key=key)

    # ── Public ────────────────────────────────────────────────────────────

    def normalize(self, text: str) -> str:
        """
        Sends the raw user text to GPT-4o-mini and gets back a clean
        Modern Standard Arabic version. Used before the local NLP model.
        """
        try:
            resp = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": _NORMALIZE_SYSTEM},
                    {"role": "user",   "content": text},
                ],
                max_tokens=120,
                temperature=0.1,
            )
            normalized = resp.choices[0].message.content.strip()
            return normalized if normalized else text
        except Exception:
            return text  # graceful degradation — return original if API fails

    def extract_scores(self, text: str) -> dict:
        """
        Extracts student's actual scores (ignoring max possible marks)
        and returns them as a clean JSON dictionary mapped to dataset columns.
        """
        import json
        try:
            resp = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": _EXTRACT_SYSTEM},
                    {"role": "user",   "content": text},
                ],
                max_tokens=100,
                temperature=0.0,
                response_format={ "type": "json_object" }
            )
            content = resp.choices[0].message.content.strip()
            return json.loads(content)
        except Exception:
            return {}

    def generate_response(
        self,
        user_text: str,
        known_scores: dict | None = None,
        intent: str | None = None,
        lang: str = "العربية"
    ) -> str:
        """
        Full fallback response in the selected language.
        Includes any known scores as context so the answer is personalised.
        """
        context_block = ""
        is_en = (lang == "English")
        
        if known_scores:
            parts = [
                f"- {col.replace('_', ' ')}: {val}"
                for col, val in known_scores.items()
            ]
            context_block = ("\n\nStudent's recorded grades:\n" if is_en else "\n\nدرجات الطالب المتاحة:\n") + "\n".join(parts)

        intent_hint = ""
        if intent == "predict_performance":
            intent_hint = "The student wants to know their chances of passing. " if is_en else "الطالب يريد معرفة فرصته في النجاح. "
        elif intent == "get_advice":
            intent_hint = "The student wants a study plan or advice. " if is_en else "الطالب يريد نصيحة أو خطة للمذاكرة. "

        if is_en:
            system_msg = "You are a helpful academic advisor for university students. Reply in English briefly and in a friendly, practical tone." + f"\n\n{intent_hint}" + context_block
        else:
            system_msg = _SYSTEM_BASE + f"\n\n{intent_hint}" + context_block

        try:
            resp = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_text},
                ],
                max_tokens=300,
                temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return (
                "😅 I'm having trouble connecting to the AI server — please try again later!" 
                if is_en else
                "😅 حصلت مشكلة في الاتصال بالذكاء الاصطناعي — حاول تاني بعد شوية!"
            )

    @classmethod
    def is_available(cls) -> bool:
        """Returns True if an API key is configured."""
        return bool(os.environ.get("OPENAI_API_KEY", ""))
