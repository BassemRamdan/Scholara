"""
Conversation Manager — orchestrates multi-turn dialogue with a pure local NLP pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ── Dataset Feature Registry ──────────────────────────────────────────────
# Each feature has a list of natural-language descriptions in AR + EN.
FEATURE_DESCRIPTIONS: dict[str, list[str]] = {
    "Midterm_Score": [
        "ميد تيرم", "midterm", "نص الترم", "اختبار منتصف الفصل",
        "half term exam", "mid exam", "درجة الميد", "امتحان الميد",
        "ميدتيرم", "الميد", "امتحان الميدتيرم", "درجات الميد",
    ],
    "Final_Score": [
        "فاينل", "final exam", "الامتحان النهائي", "نهاية الترم",
        "final test", "امتحان الفاينل", "درجة الفاينل",
        "فاينال", "الفاينال", "النهائي", "امتحان الفاينال", "درجة الفاينال",
    ],
    "Assignments_Avg": [
        "اسايمنت", "assignment", "واجب", "homework", "task",
        "الواجبات", "درجة الواجب", "assignments average",
        "واجبات", "الاسايمنت", "الاسايمنتات", "اسايمنتات",
    ],
    "Quizzes_Avg": [
        "كويز", "quiz", "اختبار قصير", "short test", "quizzes",
        "درجة الكويز", "الكويزات", "كويزات", "الكويز",
    ],
    "Participation_Score": [
        "مشاركة", "participation", "مشاركة في الفصل",
        "class participation", "engage", "درجة المشاركة",
        "مشاركه", "المشاركة", "المشاركه", "تفاعل",
    ],
    "Projects_Score": [
        "بروجكت", "project", "مشروع", "projects", "درجة البروجكت",
        "بروجكتات", "البروجكت", "المشروع",
    ],
    "Attendance (%)": [
        "حضور", "attendance", "غياب", "absent", "present",
        "نسبة الحضور", "attendance percentage", "حضرت",
        "الحضور", "نسبه الحضور", "الغياب",
    ],
}

@dataclass
class ConversationContext:
    state: str = "idle"
    pending_score: int | None = None
    known_scores: dict[str, int] = field(default_factory=dict)
    last_intent: str | None = None
    turn_count: int = 0


class ConversationManager:
    """
    Orchestrates dialogue routing using local NLP embeddings and pattern matching.
    """

    _FEATURE_MATCH_THRESHOLD = 0.45

    def __init__(self, nlp, recommender):
        self._nlp     = nlp
        self._rec     = recommender
        self._model   = nlp._model

        self._feature_embeddings: dict[str, np.ndarray] = {}
        for col, descs in FEATURE_DESCRIPTIONS.items():
            self._feature_embeddings[col] = self._model.encode(
                descs, convert_to_numpy=True
            )

    # ── Public API ────────────────────────────────────────────────────────

    def handle(
        self,
        user_text: str,
        ctx: ConversationContext,
        lang: str = "العربية",
    ) -> tuple[str, str]:
        """
        Handles text dialogue.
        Returns a tuple: (response_text, processed_user_input_display)
        """
        ctx.turn_count += 1
        is_ar = (lang == "العربية")
        user_display = user_text

        # Default fallback messages for score type clarifications
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
            res = self._resolve_score_type(user_text, ctx, lang)
            return res, user_display

        # 1. Extract scores and keywords from original raw user_text
        raw_scores = self._extract_numbers(user_text)
        raw_matched_col = self._match_feature(user_text)

        # 2. Intercept capability/feature queries about files/voice to inform user they are disabled
        lower_text = user_text.lower()
        has_file_kw = any(k in lower_text for k in ["صوره", "صورة", "ملف", "ارفق", "أرفق", "شيت", "جدول", "ورقة", "ورقه", "pdf", "image", "file", "photo", "upload", "attachment", "screenshot"])
        has_voice_kw = any(k in lower_text for k in ["صوت", "اتكلم", "تسجيل", "سجل", "ميك", "مك", "mic", "audio", "record", "voice", "speech", "listen"])
        
        if (has_file_kw or has_voice_kw) and not raw_scores:
            if has_file_kw:
                return (
                    "عذراً، لقد تم إيقاف ميزة قراءة الصور والملفات للحفاظ على بساطة وسرعة النظام. يمكنك كتابة درجاتك أو أسئلتك نصياً! 📝"
                    if is_ar else
                    "Sorry, image and file reading features have been disabled to keep the system simple and fast. Please type your grades or questions instead! 📝"
                ), user_display
            else:
                return (
                    "عذراً، ميزة التسجيل الصوتي غير مفعلة حالياً في هذه النسخة المخففة. تفضل بكتابة سؤالك! ⌨️"
                    if is_ar else
                    "Sorry, voice recording is not enabled in this streamlined version. Please type your question! ⌨️"
                ), user_display

        # 3. Process normalization and score matching
        nlp_result = self._nlp.analyze_input(user_text)
        nlp_result["original_text"] = user_text
        
        # Merge scores extracted locally from raw and normalized versions
        all_scores = raw_scores if raw_scores else nlp_result["extracted_scores"]
        nlp_result["extracted_scores"] = all_scores
        
        ctx.last_intent = nlp_result["intent"]

        matched_col = raw_matched_col or self._match_feature(user_text)
        if matched_col and all_scores:
            score_val = self._best_score_candidate(all_scores)
            if score_val is not None:
                ctx.known_scores[matched_col] = score_val

        if all_scores and not matched_col:
            ctx.pending_score = self._best_score_candidate(all_scores)
            if ctx.pending_score is not None:
                ctx.state = "awaiting_score_type"
                return clarify_msg, user_display

        # ── Route — pure local engine ───────────────
        ctx.state = "responding"
        nlp_result["known_scores"] = dict(ctx.known_scores)
        nlp_result["lang"] = lang

        # High confidence → use local recommendation engine
        res = self._rec.get_recommendation(nlp_result)
        return res, user_display

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _normalize_arabic_basic(text: str) -> str:
        """Standardizes common spelling variations in Arabic to make substring matching highly robust."""
        text = text.lower()
        text = re.sub(r"[أإآ]", "ا", text)
        text = re.sub(r"ى", "ي", text)
        text = re.sub(r"ة", "ه", text)
        return text

    def _match_feature(self, text: str) -> str | None:
        """
        Finds which dataset feature column the text matches.
        """
        clean_text = re.sub(r'[\d\s\W_]+', '', text)
        if not clean_text:
            return None

        norm_text = self._normalize_arabic_basic(text)

        # Exact Keyword/Substring matching
        for col, descs in FEATURE_DESCRIPTIONS.items():
            for desc in descs:
                norm_desc = self._normalize_arabic_basic(desc)
                if norm_desc in norm_text:
                    return col

        # Semantic Similarity Fallback
        user_emb = self._model.encode([text], convert_to_numpy=True)
        best_col = None
        best_sim = self._FEATURE_MATCH_THRESHOLD

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

        # Break out of loop naturally for conversational inputs
        ctx.state = "idle"
        ctx.pending_score = None
        res, _ = self.handle(user_text, ctx, lang)
        return res

    @staticmethod
    def _extract_numbers(text: str) -> list[int]:
        text = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
        return [int(n) for n in re.findall(r"\d+", text)]

    @staticmethod
    def _best_score_candidate(scores: list[int]) -> int | None:
        valid = [s for s in scores if 0 <= s <= 100]
        return valid[0] if valid else None
