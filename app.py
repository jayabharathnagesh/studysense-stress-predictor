import json

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="StudySense", layout="centered")

LABELS = {0: "Low", 1: "Medium", 2: "High"}
COLORS = {0: "green", 1: "orange", 2: "red"}      # used for coloured text

# ---------------------------------------------------------------------------
# Feedback shown after the result. It depends on two things:
#   1. the predicted level (0 = Low, 1 = Medium, 2 = High)
#   2. how confident the model is in that level, in three bands:
#        band 0 = 0 to 49%, band 1 = 50 to 79%, band 2 = 80 to 100%
# These are general wellbeing tips, not medical advice.
# ---------------------------------------------------------------------------
FEEDBACK = {
    0: {  # LOW
        0: {
            "title": "Leaning Low, but close to the edge",
            "message": "Your answers point toward low stress, but the model isn't very sure. "
                       "A few answers look closer to medium stress.",
            "tips": [
                "Notice which areas (sleep, workload, support, mood) feel least settled, and protect those first.",
                "Keep a regular sleep schedule, especially in exam weeks.",
                "Check in again in a week, or when your workload changes.",
            ],
        },
        1: {
            "title": "Low stress risk",
            "message": "Your answers mostly match students with low stress. "
                       "You seem to be coping well right now.",
            "tips": [
                "Keep the routines that are working: sleep, meals, movement and time with friends.",
                "Leave a small buffer before deadlines so a busy week doesn't catch you off guard.",
                "Stay connected with people you trust. Support is the best protection when things get harder.",
            ],
        },
        2: {
            "title": "Strong match with low stress",
            "message": "Your answers closely match students with low stress. "
                       "Whatever you are doing is working.",
            "tips": [
                "Keep your current habits, especially sleep and social time.",
                "Check in on a friend who might be struggling. You may have energy to spare.",
                "Remember this is a snapshot. Retake it during exam weeks or when things change.",
            ],
        },
    },
    1: {  # MEDIUM
        0: {
            "title": "Mixed signals: between two stress levels",
            "message": "The model isn't sure. Your answers sit between two stress levels, "
                       "so treat this as a nudge, not a verdict.",
            "tips": [
                "Pick the one or two answers that worry you most (for example sleep or workload) and improve those first.",
                "Talk to a friend or classmate about how the week is going.",
                "Retake the check in a week to see which way things are moving.",
            ],
        },
        1: {
            "title": "Moderate stress risk",
            "message": "Your answers match students with a medium level of stress. "
                       "It is manageable, and a good time to act before it builds.",
            "tips": [
                "Plan your week and break big tasks into small daily steps.",
                "Protect your sleep and add short daily breaks, such as a 15-minute walk.",
                "Tell a friend, mentor or teacher if your workload feels too heavy. Asking early is easier than asking late.",
            ],
        },
        2: {
            "title": "Clear signs of moderate stress",
            "message": "Your answers closely match students with medium stress. "
                       "This is a good moment to lighten the load before it climbs.",
            "tips": [
                "Drop or postpone one optional commitment this week.",
                "Set a fixed sleep window and stick to it for five days.",
                "Speak to a faculty advisor or mentor about your workload.",
                "If this feeling lasts more than two weeks, book a session with your college counsellor.",
            ],
        },
    },
    2: {  # HIGH
        0: {
            "title": "Leaning High, but not certain",
            "message": "Your answers lean toward high stress, but some look closer to medium. "
                       "Take it seriously without panicking.",
            "tips": [
                "Talk to someone you trust this week, such as a friend, family member or mentor.",
                "Choose one stressor you can reduce now and deal with that first.",
                "Protect your sleep and meals, since they affect everything else.",
                "If you feel overwhelmed, a college counsellor is a good next step.",
            ],
        },
        1: {
            "title": "High stress risk",
            "message": "Your answers match students who report high stress. It is worth acting on now.",
            "tips": [
                "Reach out to a counsellor, mentor or family member and tell them how you are doing.",
                "Ask for help with workload: an extension, a lighter schedule, or a talk with your advisor.",
                "Keep the basics steady: sleep, regular meals and a little movement each day.",
                "Cut back on anything optional for the next week or two.",
            ],
        },
        2: {
            "title": "Strong match with high stress",
            "message": "Your answers closely match students with high stress. "
                       "Please don't carry this alone.",
            "tips": [
                "Talk to a counsellor, doctor or someone you trust this week, not later.",
                "Tell one person close to you how you have been feeling.",
                "Take the pressure off where you can: postpone what is not urgent and ask for deadline help.",
                "If things ever feel unsafe or hopeless, contact a helpline or emergency services right away (see below).",
            ],
        },
    },
}

HELPLINE = (
    "**Need someone to talk to?** Tele-MANAS (Government of India) offers free, "
    "24x7 counselling by phone in 20 Indian languages: call **14416**. "
    "In an emergency, call **112**."
)


def confidence_band(pct):
    """Map the model's confidence (a whole percent) to a feedback band."""
    if pct <= 49:
        return 0
    if pct <= 79:
        return 1
    return 2


# ---------------------------------------------------------------------------
# Top reasons. Instead of the shap library (which can be blocked on locked-down
# laptops), the app asks the model directly: "how much would your confidence
# change if this one answer were replaced by a typical answer?" For each answer
# we try every value on its scale, average the model's confidence, and compare
# that with the confidence for the answers you actually gave. In a test on 220
# students this matched SHAP's #1 reason 90% of the time.
# ---------------------------------------------------------------------------
FACTOR_TIPS = {
    "anxiety_level": "Try a few minutes of slow breathing or a short walk when anxiety spikes, and talk to a counsellor if it stays high.",
    "self_esteem": "Write down three things you handled well this week, and set goals small enough to finish.",
    "mental_health_history": "A past mental health concern is nothing to be ashamed of. Keep any support or check-ins you already have going.",
    "depression": "If low mood has lasted more than two weeks, please talk to a counsellor or doctor.",
    "headache": "Frequent headaches are worth mentioning to a doctor. Water, regular meals and screen breaks can help meanwhile.",
    "blood_pressure": "This is a strong signal in this dataset, but the pattern looks unusual, so don't read much into it. If you are worried about your blood pressure, get it checked.",
    "sleep_quality": "Keep a fixed sleep and wake time, and put screens away for 30 minutes before bed.",
    "breathing_problem": "Breathing difficulties deserve a medical check-up. Don't assume it is only stress.",
    "noise_level": "Try a quieter study spot, earplugs or headphones, or the library.",
    "living_conditions": "Small changes help: tidy a study corner, and ask your hostel warden or landlord about problems you can't fix yourself.",
    "safety": "If you feel unsafe where you live or study, tell someone you trust or your college authorities. Your safety comes first.",
    "basic_needs": "If food, money or supplies are a worry, ask your college about support schemes and talk to someone you trust.",
    "academic_performance": "Pick one subject to improve first and ask a teacher or classmate for help. Small steady steps count.",
    "study_load": "List everything that is due, then drop, delay or share what you can. Study in short blocks with breaks.",
    "teacher_student_relationship": "Have a short conversation with one teacher or mentor about where you are stuck. Many are glad to help.",
    "future_career_concerns": "Write down one small next step, such as a skill to learn or a person to ask. Your placement cell or a mentor can help.",
    "social_support": "Reach out to one friend or family member this week, even for a short chat. A club or study group can help too.",
    "peer_pressure": "It is okay to say no. Spend time with friends who respect your choices, and talk to a mentor if the pressure is heavy.",
    "extracurricular_activities": "Activities can help or add load. Keep the ones you enjoy and pause the ones that feel like a burden.",
    "bullying": "Bullying is not your fault. Tell a trusted person and your college's grievance or anti-ragging cell, and keep a record of what happens.",
}


def top_reasons(model, features, values, level, n=3):
    """Return up to n (feature, effect) pairs: the answers that raised the model's
    confidence in `level` the most. effect is a fraction (0.10 = 10 points)."""
    cols = list(features)
    ci = list(model.classes_).index(level)
    p_now = model.predict_proba(pd.DataFrame([values])[cols])[0][ci]

    rows, spans = [], []
    for c in cols:
        start = len(rows)
        for v in range(features[c]["min"], features[c]["max"] + 1):
            row = dict(values)
            row[c] = v
            rows.append(row)
        spans.append((c, start, len(rows)))

    probs = model.predict_proba(pd.DataFrame(rows)[cols])[:, ci]
    effect = {c: float(p_now - probs[s:e].mean()) for c, s, e in spans}
    ranked = sorted(effect.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return [(c, e) for c, e in ranked if e >= 0.01]     # ignore effects under 1 point


def describe_answer(value, info):
    if info["min"] == 0 and info["max"] == 1:
        return "Yes" if value == 1 else "No"
    return f"{value} (scale {info['min']} to {info['max']})"


@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    with open("features.json") as f:
        features = json.load(f)
    return model, features


model, features = load_model()

st.title("StudySense")
st.subheader("Student stress risk indicator")
st.write(
    "Answer the questions below to see an estimated stress risk level. "
    "**This is a risk indicator, not a medical diagnosis.** "
    "If you are struggling, please talk to someone you trust or a qualified professional."
)

# One input per feature the model was trained on
values = {}
for name, info in features.items():
    label = name.replace("_", " ").capitalize()
    if info["min"] == 0 and info["max"] == 1:
        choice = st.radio(label, ["No", "Yes"], horizontal=True)
        values[name] = 1 if choice == "Yes" else 0
    else:
        values[name] = st.slider(label, info["min"], info["max"], info["default"])

if st.button("Check my risk"):
    # Keep the columns in the same order the model was trained on
    row = pd.DataFrame([values])[list(features.keys())]

    level = int(model.predict(row)[0])
    probs = {int(c): float(p) for c, p in zip(model.classes_, model.predict_proba(row)[0])}

    # Whole percentages, used for the bars AND for choosing the feedback band,
    # so the number you see always matches the message you get.
    pcts = {c: round(p * 100) for c, p in probs.items()}
    top_pct = pcts[level]
    feedback = FEEDBACK[level][confidence_band(top_pct)]

    st.subheader(f"Estimated stress risk: :{COLORS[level]}[{LABELS[level]}] ({top_pct}% confidence)")

    # Progress bars instead of st.bar_chart: they work even where the pyarrow
    # library is blocked (for example by a college or work security policy).
    st.write("How confident the model is in each level:")
    for c, pct in pcts.items():
        st.progress(pct / 100, text=f":{COLORS[c]}[**{LABELS[c]}**]: {pct}%")

    # Green for Low, yellow for Medium, red for High
    box = {0: st.success, 1: st.warning, 2: st.error}[level]
    box(f"**{feedback['title']}**\n\n{feedback['message']}")
    st.markdown("**General tips**\n\n" + "\n".join(f"- {tip}" for tip in feedback["tips"]))

    # Top 3 reasons, with a tip for each when the level is Medium or High
    reasons = top_reasons(model, features, values, level)
    if reasons:
        st.markdown("**Top reasons for this result**")
        lines = []
        for k, (name, effect) in enumerate(reasons, start=1):
            label = name.replace("_", " ").capitalize()
            answer = describe_answer(values[name], features[name])
            line = (f"{k}. **{label}**: your answer was {answer}, "
                    f"adding about {round(effect * 100)} points toward {LABELS[level]}.")
            if level in (1, 2):
                line += f"\n    - *What may help:* {FACTOR_TIPS[name]}"
            lines.append(line)
        st.markdown("\n".join(lines))
        st.caption(
            "Points show how much each answer raised the model's confidence in this level, "
            "compared with a typical answer on the same scale."
        )

    if level == 2:
        st.info(HELPLINE)

st.caption(
    "The percentage shows how confident the model is in its answer, not how severe your stress is. "
    "The tips are general wellbeing suggestions, not medical advice. "
    "Built on a public Kaggle student stress dataset that may be synthetic, so predictions show "
    "patterns in that data, not real-world causes."
)
