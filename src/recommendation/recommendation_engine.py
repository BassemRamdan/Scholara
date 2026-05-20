"""
Recommendation Engine — generates premium, highly detailed academic advising reports,
study plans, and performance predictions in Egyptian Arabic or English.
Runs 100% offline and locally.
"""

from __future__ import annotations
import random

# ── Detailed Advising Templates (Arabic / Egyptian) ───────────────────────────
_PASS_HIGH_AR = [
    "🎉 **عاش يا بطل! درجتك دي بتثبت إنك ماشي على الطريق الصح وبقوة!**\n\n"
    "💪 مستواك متميز جداً وفي منطقة الأمان التام. طموحك دلوقتي لازم يكون الامتياز والترتيب على الدفعة!\n\n"
    "🏆 **توصياتي الأكاديمية عشان تحافظ على التفوق ده:**\n"
    "- 🧠 **طوّر فهمك:** جرب تشرح النقط الصعبة لزمايلك؛ الشرح وتوصيل المعلومة بيرسخ الفهم في دماغك بنسبة 90%.\n"
    "- 🚀 **شارك في مشاريع:** استغل وقت فراغك في تطبيق اللي بتدرسه عملياً، ده هيقوي الـ Resume بتاعك جداً.\n"
    "- 🎯 **حافظ على الشغف:** وازن بين مذاكرتك وصحتك النفسية، التفوق رحلة مستمرة مش يوم واحد.",
    
    "⭐ **درجة ممتازة وتستحق كل تقدير! أنت بجد فخر لنفسك ولدكاترتك!**\n\n"
    "ركز في الفترات الجاية إنك تقفل الفاينل بكل ثقة وتضمن التقدير العالي.\n\n"
    "💡 **نصائح للتفوق المستمر:**\n"
    "- 📚 **توسيع المدارك:** اقرأ مراجع إضافية أو كورسات خفيفة متعلقة بموادك لتكتسب ميزة تنافسية.\n"
    "- 🔍 **تحليل الأخطاء:** حتى لو ناقص درجات بسيطة، اعرف نقصتها ليه عشان متكررهاش في الفاينل."
]

_PASS_MID_AR = [
    "😊 **مستواك كويس وجيد جداً! أنت في المنطقة الآمنة بس تقدر تطور أكتر بكتير!**\n\n"
    "شوية مجهود صغيرين هينقلوك لمنطقة الامتياز. الفاينل هو فرصتك الذهبية عشان تثبت نفسك وتفاجئ الكل!\n\n"
    "🎯 **خطتي المقترحة ليك حالياً:**\n"
    "- 📅 **جدول المذاكرة:** حدد ساعتين يومياً ثابتين للمذاكرة والمراجعة والتزم بيهم تماماً.\n"
    "- 🔄 **طريقة الطماطم (Pomodoro):** ذاكر 25 دقيقة بتركيز كامل، وخود 5 دقائق راحة؛ الطريقة دي بتمنع التشتت وتزيد الإنتاجية.\n"
    "- 📝 **التلخيص الفعال:** اكتب ملخصاتك بنفسك وبألوان مختلفة، ده بيسهل استرجاع المعلومة في الامتحان.",
    
    "📚 **درجة كويسة جداً وبتأكد إن عندك أساس متين تقدر تبني عليه!**\n\n"
    "محتاج بس ترتب أفكارك وتشد حيلك سيكا عشان تضمن الـ A في الفاينل.\n\n"
    "💡 **نصائح عملية سريعة:**\n"
    "- 🤝 **المذاكرة الثنائية:** ذاكر مع زميل شاطر وملتزم، شجعوا بعض وحلوا مسائل مع بعض.\n"
    "- ⏳ **حل امتحانات سابقة:** ده أفضل أسلوب لكسر رهبة الامتحان ومعرفة أهم الأجزاء اللي الدكتور بيركز عليها."
]

_FAIL_RISK_AR = [
    "⚠️ **يا بطل، الدرجة دي بتقول إننا في خطر ومحتاجين نتحرك فوراً! بس لسه الفرصة في إيدينا!**\n\n"
    "😟 المستوى ده يعتبر منخفض ومقلق، ومحتاجين نركز ونعوض في أعمال السنة الجاية والامتحان النهائي. ماتقلقش ولا تيأس، في طلاب كتير كانوا في نفس مكانك وشدو حيلهم ونجحوا بتقديرات ممتازة!\n\n"
    "🛠️ **خطة الإنقاذ الأكاديمية العاجلة:**\n"
    "1️⃣ **تحديد نقاط الضعف:** اعمل قائمة فورية بالمواضيع اللي مش فاهمها أو حاسس إنك ضايع فيها.\n"
    "2️⃣ **التخلص من التراكمات:** خصص ساعة واحدة يومياً لمراجعة القديم بجانب تتبع المحاضرات الجديدة أول بأول.\n"
    "3️⃣ **استعن بغيرك:** اسأل المعيد في الساعات المكتبية، أو اطلب مساعدة زمايلك الشاطرين؛ متتكسفش أبداً!\n"
    "4️⃣ **التدريب العملي:** حل شيتات ومسائل بإيدك، القراءة البصرية لوحدها مش كفاية في المواد الأكاديمية.",
    
    "🚨 **وقف هنا ثانية.. الدرجة دي جرس إنذار ومحتاجة وقفة جدية مع النفس!**\n\n"
    "مستواك الحالي محتاج خطة تعويضية صارمة وسريعة. لسه في وقت كافي للتعويض والنجاح، بس البداية لازم تكون من النهاردة مش بكره!\n\n"
    "💡 **خطوات عملية فورية:**\n"
    "- 📴 **منع المشتتات:** اقفل السوشيال ميديا تماماً وقت المذاكرة؛ التركيز الصافي هو مفتاح النجاح السريع.\n"
    "- 😴 **تنظيم النوم:** السهر الكتير بيضعف الذاكرة والتركيز. اضبط نومك (6-8 ساعات ليلاً) عشان تستوعب كويس.\n"
    "- 📚 **التركيز على الأساسيات:** افهم العناوين الرئيسية والقواعد الأساسية للمادة الأول، وبعدين ادخل في التفاصيل."
]

_ADVICE_GENERAL_AR = [
    "📝 **نصيحتي الأكاديمية:** نظم وقتك واعمل جدول أسبوعي متوازن، وخصص وقت كافي للراحة والترفيه عشان متفقدش شغفك.",
    "🎯 **مفتاح النجاح الأكبر:** حل أسئلة كتير وبأفكار متنوعة، التكرار بيعلم الشطار وبيثبت المعلومة بنسبة 100%.",
    "🤝 **التعاون الدراسي:** المذاكرة الجماعية مع شلة ملتزمة بتشجع وبتفتح آفاق جديدة للفهم والمناقشة."
]

# ── Detailed Advising Templates (English) ─────────────────────────────────────
_PASS_HIGH_EN = [
    "🎉 **Fantastic! Your score is outstanding and shows complete dedication!**\n\n"
    "💪 You are in the top tier and completely safe. Your goal now should be achieving the highest grade (A+)!\n\n"
    "🏆 **Academic Recommendations for Excellence:**\n"
    "- 🧠 **Teach to Learn:** Explain complex concepts to your peers. Teaching cements knowledge up to 90%.\n"
    "- 🚀 **Apply Your Skills:** Engage in extra projects or apply your coursework to real-world scenarios.\n"
    "- 🎯 **Maintain Balance:** Keep up the great routine and take care of your physical and mental well-being.",
    
    "⭐ **Superb score! You have proven that you possess an exceptional grasp of the material!**\n\n"
    "Keep this momentum going into the final exams to secure your top ranking."
]

_PASS_MID_EN = [
    "😊 **Solid performance! You are in the safe zone but have the potential to go much higher!**\n\n"
    "With a little extra effort, you can easily shift into the A-range. The finals are your playground to shine!\n\n"
    "🎯 **Your Academic Action Plan:**\n"
    "- 📅 **Establish a Routine:** Dedicate at least 2 hours of focused daily study and stick to it.\n"
    "- 🔄 **Pomodoro Technique:** Study for 25 minutes, then take a 5-minute break. This prevents mental fatigue.\n"
    "- 📝 **Smart Summaries:** Write summaries or mind maps of lectures using colors to aid memory retention."
]

_FAIL_RISK_EN = [
    "⚠️ **Let's take a serious look here. This grade indicates you are at risk, but there is still plenty of time to turn it around!**\n\n"
    "😟 This score is below the safe passing threshold. You need to act immediately and compensate in upcoming assignments and the final exam. Don't worry, many students have successfully bounced back from this position!\n\n"
    "🛠️ **Your Step-by-Step Recovery Plan:**\n"
    "1️⃣ **List Your Weak Spots:** Write down the exact chapters/topics where you feel lost.\n"
    "2️⃣ **Eliminate Backlogs:** Devote 1 hour daily to catching up on past lessons while keeping up with new ones.\n"
    "3️⃣ **Ask for Help:** Don't hesitate to visit your teaching assistant during office hours or ask classmates.\n"
    "4️⃣ **Practice Active Recall:** Solve exercises and quiz yourself rather than just highlighting text.",
    
    "🚨 **Academic Alert: This score is a wake-up call! Let's build a strategy to secure your success.**\n\n"
    "Your current standing requires a disciplined and immediate recovery plan. You can still pass with a decent grade, but the work starts today!\n\n"
    "💡 **Immediate Strategies:**\n"
    "- 📴 **Limit Distractions:** Put your phone in another room while studying to ensure maximum focus.\n"
    "- 😴 **Optimize Sleep:** Sleeping 7-8 hours is crucial for cognitive function. Do not compromise sleep for cramming.\n"
    "- 📚 **Master the Core:** Focus on fundamental concepts first before trying to learn advanced nuances."
]

_ADVICE_GENERAL_EN = [
    "📝 **Key Advice:** Maintain a balanced daily study schedule and remember to allocate time for active rest to avoid burnout.",
    "🎯 **The Secret to Success:** Solve past exams and mock tests. Practice builds familiarity with exam patterns."
]


class RecommendationEngine:
    def get_recommendation(self, nlp_result: dict) -> str:
        intent = nlp_result.get("intent", "get_advice")
        scores = nlp_result.get("extracted_scores", [])
        known_scores = nlp_result.get("known_scores", {})
        lang = nlp_result.get("lang", "العربية")

        if nlp_result.get("is_greeting"):
            if lang == "English":
                return "👋 Hello! I'm EduGuide, your smart local academic advisor. How can I help you today? You can ask me about your grades, study plans, or ask for guidance! 🎓✨"
            else:
                return "👋 أهلاً بك يا بطل! أنا EduGuide، مساعدك الأكاديمي الذكي والمحلي. كيف يمكنني مساعدتك اليوم؟ اسألني عن درجاتك، خطط المذاكرة، أو اطلب نصيحة عامة! 🎓✨"

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
            def format_score(col, val):
                mx = 100
                if col == "Midterm_Score": mx = 30
                elif col == "Final_Score": mx = 50
                elif col == "Participation_Score": mx = 10
                return f"- **{col.replace('_', ' ')}**: {val} من {mx}" if not is_en else f"- **{col.replace('_', ' ')}**: {val}/{mx}"
                
            cols = "\n".join(format_score(col, val) for col, val in known_scores.items())
            context_note = ("\n\n📊 **درجاتك المسجلة:**\n" if not is_en else "\n\n📊 **Your Recorded Scores:**\n") + cols

        if score >= 80:
            base = random.choice(_PASS_HIGH_EN if is_en else _PASS_HIGH_AR)
        elif score >= 60:
            base = random.choice(_PASS_MID_EN if is_en else _PASS_MID_AR)
        else:
            base = random.choice(_FAIL_RISK_EN if is_en else _FAIL_RISK_AR)

        score_label = "Score (%)" if is_en else "النسبة المئوية للدرجة (%)"
        return f"{base}\n\n📍 *(تحليل مبني على {score_label}: **{score}**)*{context_note}"

    # ── Advice response ───────────────────────────────────────────────────
    def _advise(self, score: int | None, lang: str) -> str:
        is_en = (lang == "English")
        
        if score is not None and score >= 80:
            return random.choice(_PASS_HIGH_EN if is_en else _PASS_HIGH_AR)
        elif score is not None and score < 60:
            return random.choice(_FAIL_RISK_EN if is_en else _FAIL_RISK_AR)
        elif score is not None:
            return random.choice(_PASS_MID_EN if is_en else _PASS_MID_AR)
        else:
            return random.choice(_ADVICE_GENERAL_EN if is_en else _ADVICE_GENERAL_AR)

    @staticmethod
    def _best_score(scores: list, known_scores: dict) -> int | None:
        if known_scores:
            tot_achieved = 0.0
            tot_possible = 0.0
            
            if "Midterm_Score" in known_scores:
                tot_achieved += known_scores["Midterm_Score"]
                tot_possible += 30
            if "Final_Score" in known_scores:
                tot_achieved += known_scores["Final_Score"]
                tot_possible += 50
            if "Assignments_Avg" in known_scores:
                tot_achieved += known_scores["Assignments_Avg"] * 0.1
                tot_possible += 10
            if "Quizzes_Avg" in known_scores:
                tot_achieved += known_scores["Quizzes_Avg"] * 0.1
                tot_possible += 10
                
            # Projects and Participation do not factor into the out-of-100 total score of the dataset
            # so we only base percentage on the parts that do.
            if tot_possible > 0:
                return int((tot_achieved / tot_possible) * 100)
                
        valid = [s for s in scores if 0 <= s <= 100]
        return valid[0] if valid else None
