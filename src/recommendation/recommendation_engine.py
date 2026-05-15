"""
Recommendation Engine — generates Egyptian Arabic or English responses based on
NLP intent + conversation context.
"""

from __future__ import annotations
import random


# ── Response banks (Arabic) ──────────────────────────────────────────────────
_PASS_HIGH_AR = [
    "🎉 بالظبط زي ما توقعت — درجتك دي قوية جداً وأنت في مكان ممتاز! كمّل على نفس الأسلوب.",
    "💪 درجتك فوق المتوسط بشكل واضح، بنسبة كبيرة هتعدي ومحتاج بس تحافظ على المستوى ده.",
]
_PASS_MID_AR = [
    "😊 درجتك كويسة وفي المنطقة الآمنة، بس متسيبش نفسك — شوية مجهود زيادة هيضمنلك النجاح براحة.",
    "📚 مش وحشة خالص! بس لو ذاكرت شوية أكتر للفاينل، هتبقى في بر الأمان تماماً.",
]
_FAIL_RISK_AR = [
    "⚠️ الرقم ده بيقولي إن في خطر شوية — بس لسه الفرصة موجودة! محتاج تشد حيلك من النهارده.",
    "😟 الدرجة دي تعتبر منخفضة، ومحتاج تعوضها قبل الفاينل. ماتيأسش — في وقت لسه!",
]
_ADVICE_GENERAL_AR = [
    "📝 نصيحتي: اعمل جدول مذاكرة واثبت عليه، والتزم بيه حتى لو ساعة في اليوم.",
    "🎯 حل أسئلة كتير من السنين الجاية — ده أقوى أسلوب للمذاكرة.",
    "😴 النوم الكافي مهم زي المذاكرة بالظبط — ماتسهرش على حساب صحتك.",
    "🤝 حاول تذاكر مع زميل — شرح المعلومة لحد تاني بيرسّخها أكتر في دماغك.",
]
_ADVICE_HIGH_SCORE_AR = [
    "⭐ ممتاز! مستواك عالي — حاول تساعد زمايلك وتشرحلهم، ده بيقوّيك أكتر.",
    "🏆 أنت في المسار الصح. ركّز على النقط الضعيفة اللي ممكن تنقص منك درجات.",
]
_ADVICE_LOW_SCORE_AR = [
    "💡 أول خطوة: اعمل ليستة بالمواضيع اللي مش فاهمها، وابدأ بيها على طول.",
    "🔄 حاول تغيّر أسلوب المذاكرة — لو الطريقة القديمة مش شاغلة، جرّب تعمل ملخصات أو فيديوهات.",
]

# ── Response banks (English) ─────────────────────────────────────────────────
_PASS_HIGH_EN = [
    "🎉 Exactly what I expected! This is a very strong score. Keep up the good work.",
    "💪 Clearly above average. You're very likely to pass, just maintain this level.",
]
_PASS_MID_EN = [
    "😊 Good score! You are in the safe zone, but a little extra effort will guarantee your success.",
    "📚 Not bad at all! Just study a bit more for the finals to be completely safe.",
]
_FAIL_RISK_EN = [
    "⚠️ This score suggests you are at some risk, but there is still time! You need to push hard starting today.",
    "😟 This is quite low, and you'll need to compensate for it in the final exam. Don't give up!",
]
_ADVICE_GENERAL_EN = [
    "📝 My advice: Create a consistent study schedule and stick to it, even for an hour a day.",
    "🎯 Solve plenty of past papers — it's the most effective way to prepare.",
    "😴 Getting enough sleep is just as important as studying. Don't sacrifice your health.",
    "🤝 Try studying with a partner. Explaining concepts to someone else cements them in your memory.",
]
_ADVICE_HIGH_SCORE_EN = [
    "⭐ Excellent! Your level is very high. Try helping out classmates; teaching strengthens your own understanding.",
    "🏆 You are on the right track. Focus on any minor weak points that might cost you marks.",
]
_ADVICE_LOW_SCORE_EN = [
    "💡 First step: Make a list of the topics you don't understand and start tackling them immediately.",
    "🔄 Try changing your study methods. If reading isn't working, try writing summaries or watching video tutorials.",
]


class RecommendationEngine:
    def get_recommendation(self, nlp_result: dict) -> str:
        intent = nlp_result.get("intent", "get_advice")
        scores = nlp_result.get("extracted_scores", [])
        known_scores = nlp_result.get("known_scores", {})
        lang = nlp_result.get("lang", "العربية")

        score = self._best_score(scores, known_scores)

        if intent == "predict_performance":
            return self._predict(score, known_scores, lang)
        else:
            return self._advise(score, lang)

    def _predict(self, score: int | None, known_scores: dict, lang: str) -> str:
        is_en = (lang == "English")
        if score is None:
            if is_en:
                return (
                    "🤔 I want to help predict your performance, "
                    "but I need to know your grades first!\n"
                    "For example, tell me: *My midterm score is 65*"
                )
            else:
                return (
                    "🤔 عايز أساعدك تعرف هتنجح ولا لأ، "
                    "بس محتاج أعرف درجاتك الأول!\n"
                    "قولي مثلاً: *درجتي في الميد تيرم 65*"
                )

        context_note = ""
        if known_scores:
            cols = ", ".join(
                f"{col.replace('_', ' ')}: **{val}**"
                for col, val in known_scores.items()
            )
            context_note = f"\n\n📊 {'Recorded' if is_en else 'اللي سجلته'}: {cols}"

        if score >= 80:
            base = random.choice(_PASS_HIGH_EN if is_en else _PASS_HIGH_AR)
        elif score >= 60:
            base = random.choice(_PASS_MID_EN if is_en else _PASS_MID_AR)
        else:
            base = random.choice(_FAIL_RISK_EN if is_en else _FAIL_RISK_AR)

        return f"{base} ({'Score' if is_en else 'الدرجة'}: **{score}**){context_note}"

    # ── Advice response ───────────────────────────────────────────────────
    def _advise(self, score: int | None, lang: str) -> str:
        is_en = (lang == "English")
        
        if score is not None and score >= 80:
            return random.choice(_ADVICE_HIGH_SCORE_EN if is_en else _ADVICE_HIGH_SCORE_AR)
        elif score is not None and score < 60:
            return random.choice(_ADVICE_LOW_SCORE_EN if is_en else _ADVICE_LOW_SCORE_AR) + "\n\n" + random.choice(_ADVICE_GENERAL_EN if is_en else _ADVICE_GENERAL_AR)
        else:
            return random.choice(_ADVICE_GENERAL_EN if is_en else _ADVICE_GENERAL_AR)

    @staticmethod
    def _best_score(scores: list, known_scores: dict) -> int | None:
        if known_scores:
            vals = list(known_scores.values())
            academic = [v for v in vals if 0 <= v <= 100]
            if academic:
                return academic[0]
        valid = [s for s in scores if 0 <= s <= 100]
        return valid[0] if valid else None
