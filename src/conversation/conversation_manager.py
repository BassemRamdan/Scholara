"""
Conversation Manager — uses semantic similarity (same multilingual embeddings)
to map user input to dataset features, instead of keyword dictionaries.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    from src.ai.openai_bridge import OpenAIBridge
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


# ── Dataset Feature Registry ──────────────────────────────────────────────
# Each feature has a list of natural-language descriptions in AR + EN.
# These are encoded once at init, then compared to user input via cosine sim.
FEATURE_DESCRIPTIONS: dict[str, list[str]] = {
    "Midterm_Score": [
        "ميد تيرم", "midterm", "نص الترم", "اختبار منتصف الفصل",
        "half term exam", "mid exam", "درجة الميد", "امتحان الميد",
    ],
    "Final_Score": [
        "فاينل", "final exam", "الامتحان النهائي", "نهاية الترم",
        "final test", "امتحان الفاينل", "درجة الفاينل",
    ],
    "Assignments_Avg": [
        "اسايمنت", "assignment", "واجب", "homework", "task",
        "الواجبات", "درجة الواجب", "assignments average",
    ],
    "Quizzes_Avg": [
        "كويز", "quiz", "اختبار قصير", "short test", "quizzes",
        "درجة الكويز", "الكويزات",
    ],
    "Participation_Score": [
        "مشاركة", "participation", "مشاركة في الفصل",
        "class participation", "engage", "درجة المشاركة",
    ],
    "Projects_Score": [
        "بروجكت", "project", "مشروع", "projects", "درجة البروجكت",
    ],
    "Attendance (%)": [
        "حضور", "attendance", "غياب", "absent", "present",
        "نسبة الحضور", "attendance percentage", "حضرت",
    ],
}

CLARIFY_SCORE_TYPE_MSG = (
    "🤔 لقيت رقم في كلامك — بس مش فاهم ده درجة إيه بالظبط؟\n\n"
    "قولي من دول:\n"
    "📝 **ميد تيرم** · 📋 **فاينل** · 📚 **اسايمنت**\n"
    "❓ **كويز** · 🎯 **بروجكت** · 📅 **حضور**\n\n"
    "_مثال: «درجتي في الميد تيرم 65»_"
)


@dataclass
class ConversationContext:
    state: str = "idle"
    pending_score: int | None = None
    known_scores: dict[str, int] = field(default_factory=dict)
    last_intent: str | None = None
    turn_count: int = 0


class ConversationManager:
    """
    Orchestrates multi-turn dialogue with a hybrid AI pipeline:

      1. OpenAI normalizes messy / Arabizi / mixed-language input
         into clean Arabic (pre-processor).
      2. Local sentence-transformer detects intent via cosine similarity.
      3. If local confidence >= 0.45 → local RecommendationEngine responds.
      4. If local confidence < 0.45  → OpenAI generates the reply directly.
    """

    _FEATURE_MATCH_THRESHOLD = 0.45
    _LOCAL_CONFIDENCE_MIN    = 0.45   # below this → OpenAI fallback

    def __init__(self, nlp, recommender, openai_bridge=None):
        self._nlp     = nlp
        self._rec     = recommender
        self._openai  = openai_bridge   # may be None if no API key
        self._model   = nlp._model

        self._feature_embeddings: dict[str, np.ndarray] = {}
        for col, descs in FEATURE_DESCRIPTIONS.items():
            self._feature_embeddings[col] = self._model.encode(
                descs, convert_to_numpy=True
            )

    # ── Public API ────────────────────────────────────────────────────────

    def handle(self, user_text: str, ctx: ConversationContext, lang: str = "العربية") -> str:
        ctx.turn_count += 1

        clarify_msg = (
            "🤔 I found a number, but I'm not sure which subject it's for.\n"
            "Please tell me: **Midterm**, **Final**, **Assignment**, **Quiz**, **Project**, or **Attendance**?\n"
            "_Example: 'My midterm score is 65'_"
        ) if lang == "English" else (
            "🤔 لقيت رقم في كلامك — بس مش فاهم ده درجة إيه بالظبط؟\n\n"
            "قولي من دول:\n"
            "📝 **ميد تيرم** · 📋 **فاينل** · 📚 **اسايمنت**\n"
            "❓ **كويز** · 🎯 **بروجكت** · 📅 **حضور**\n\n"
            "_مثال: «درجتي في الميد تيرم 65»_"
        )

        # Resolve pending clarification
        if ctx.state == "awaiting_score_type":
            return self._resolve_score_type(user_text, ctx, lang)

        normalized_text = user_text
        if self._openai:
            normalized_text = self._openai.normalize(user_text)

        nlp_result = self._nlp.analyze_input(normalized_text)
        nlp_result["original_text"] = user_text
        scores = nlp_result["extracted_scores"]
        ctx.last_intent = nlp_result["intent"]
        local_confidence = nlp_result["confidence"]

        if self._openai:
            extracted_dict = self._openai.extract_scores(normalized_text)
            text_no_digits = re.sub(r'\d+', '', normalized_text).strip()
            if len(text_no_digits) < 2 and extracted_dict:
                extracted_dict = {}

            if extracted_dict:
                ctx.known_scores.update(extracted_dict)
                ctx.pending_score = None
            elif scores:
                ctx.pending_score = self._best_score_candidate(scores)
                if ctx.pending_score is not None:
                    ctx.state = "awaiting_score_type"
                    return clarify_msg
        else:
            matched_col = self._match_feature(normalized_text)
            if matched_col and scores:
                score_val = self._best_score_candidate(scores)
                if score_val is not None:
                    ctx.known_scores[matched_col] = score_val

            if scores and not matched_col:
                ctx.pending_score = self._best_score_candidate(scores)
                if ctx.pending_score is not None:
                    ctx.state = "awaiting_score_type"
                    return clarify_msg

        # ── Step 4: Route — local engine OR OpenAI fallback ───────────────
        ctx.state = "responding"
        nlp_result["known_scores"] = dict(ctx.known_scores)
        nlp_result["lang"] = lang

        if self._openai and local_confidence < self._LOCAL_CONFIDENCE_MIN:
            # Low confidence → let OpenAI handle it
            return self._openai.generate_response(
                user_text=user_text,
                known_scores=ctx.known_scores,
                intent=ctx.last_intent,
                lang=lang
            )

        # High confidence → use local recommendation engine
        return self._rec.get_recommendation(nlp_result)

    # ── Private helpers ───────────────────────────────────────────────────

    def _match_feature(self, text: str) -> str | None:
        """
        Encodes the user's message and finds which dataset feature column
        it is semantically closest to (above threshold).
        Returns the column name or None.
        """
        user_emb = self._model.encode([text], convert_to_numpy=True)

        best_col = None
        best_sim = self._FEATURE_MATCH_THRESHOLD  # min bar to count as a match

        for col, feat_embs in self._feature_embeddings.items():
            sims = cosine_similarity(user_emb, feat_embs)[0]
            max_sim = float(np.max(sims))
            if max_sim > best_sim:
                best_sim = max_sim
                best_col = col

        return best_col

    def _resolve_score_type(self, user_text: str, ctx: ConversationContext, lang: str) -> str:
        """User is answering our clarifying question — use similarity to identify feature."""
        matched_col = self._match_feature(user_text)

        # Also extract any number from this follow-up
        extracted = self._extract_numbers(user_text)
        score_val = (
            self._best_score_candidate(extracted)
            if extracted
            else ctx.pending_score
        )

        if matched_col and score_val is not None:
            ctx.known_scores[matched_col] = score_val
            ctx.pending_score = None
            ctx.state = "responding"

            nlp_result = {
                "intent": ctx.last_intent or "predict_performance",
                "confidence": 0.9,
                "extracted_scores": list(ctx.known_scores.values()),
                "text": user_text,
                "known_scores": dict(ctx.known_scores),
                "lang": lang
            }
            return self._rec.get_recommendation(nlp_result)

        return (
            "😕 Sorry, I couldn't understand which subject you mean.\n"
            "Please specify: **Midterm**, **Final**, or **Assignment**."
        ) if lang == "English" else (
            "😕 مش قادر أفهم نوع الدرجة.\n"
            "قولي مثلاً: **ميد تيرم** أو **فاينل** أو **اسايمنت**!"
        )

    @staticmethod
    def _extract_numbers(text: str) -> list[int]:
        text = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
        return [int(n) for n in re.findall(r"\d+", text)]

    @staticmethod
    def _best_score_candidate(scores: list[int]) -> int | None:
        valid = [s for s in scores if 0 <= s <= 100]
        return valid[0] if valid else None
