import os
import re
import json
import uuid
import random
from io import BytesIO
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

from google import genai
from google.genai import types

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def load_users():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(BASE_DIR, "users.csv")

        if not os.path.exists(file_path):
            df = pd.DataFrame(columns=["login_id", "role", "name", "class_group", "linked_student_id"])
            df.to_csv(file_path, index=False)
            return df

        return pd.read_csv(file_path)

    except Exception as e:
        import streamlit as st
        st.error(f"Error loading users: {e}")
        return pd.DataFrame(columns=["login_id", "role", "name", "class_group", "linked_student_id"])
# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Handwriting AI",
    page_icon="✍️",
    layout="wide",
)

# =========================================================
# 🌈 GLOBAL SCHOOL FRIENDLY UI (FIXED + UPGRADED)
# =========================================================
st.markdown("""
<style>

/* Background gradient */
.stApp {
    background: linear-gradient(135deg, #eef2ff, #e0f2fe, #f0fdf4);
    background-size: 200% 200%;
    animation: gradientFlow 10s ease infinite;
}

@keyframes gradientFlow {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* TEXT */
body, p, div, label {
    color: #111827 !important;
}

/* INPUT FIX */
.stTextInput input {
    background: #ffffff !important;
    color: #111827 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
}

/* SELECT BOX FIX (THIS FIXES DARK DROPDOWN ISSUE) */
div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #111827 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
}

/* Dropdown menu */
ul[role="listbox"] {
    background-color: #ffffff !important;
    color: #111827 !important;
}

ul[role="listbox"] li {
    background-color: #ffffff !important;
    color: #111827 !important;
}

ul[role="listbox"] li:hover {
    background-color: #e0f2fe !important;
}

/* BUTTON */
.stButton button {
    background: linear-gradient(90deg, #6366f1, #06b6d4);
    color: white !important;
    border-radius: 10px !important;
    font-weight: 600;
}

/* CARDS */
.section-box {
    background: white;
    padding: 18px;
    border-radius: 16px;
    margin-bottom: 18px;
    box-shadow: 0 6px 14px rgba(0,0,0,0.05);
}

.result-card {
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 6px;
}

.result-green { background: #dcfce7; }
.result-red { background: #fee2e2; }
.result-blue { background: #dbeafe; }
.result-purple { background: #f3e8ff; }
.result-orange { background: #fed7aa; }

/* TITLE */
.main-title {
    font-size: 32px;
    font-weight: 800;
    text-align: center;
}

.sub-title {
    text-align: center;
    color: gray;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# FILES
# =========================================================
DATA_FILE = "results.csv"
USERS_FILE = "users.csv"
UPLOAD_DIR = "uploaded_images"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# =========================================================
# GEMINI
# =========================================================
API_KEY = st.secrets.get("GEMINI_API_KEY")

if not API_KEY:
    st.error("Add GEMINI_API_KEY in secrets.toml")
    st.stop()

client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-flash"

# =========================================================
# SESSION STATE
# =========================================================
def init_session():
    defaults = {
        "logged_in": False,
        "role": None,
        "login_id": None,
        "name": None,
        "system_id": None,
        "class_group": None,
        "linked_student_id": None,
        "otp": None,
        "otp_login_id": None,
        "otp_role": None,
        "last_analysis_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# =========================================================
# RESULTS STORAGE
# =========================================================
def ensure_results_file():
    if os.path.exists(DATA_FILE):
        return

    pd.DataFrame(columns=[
        "datetime",
        "viewer_role",
        "login_id",
        "student_name",
        "system_id",
        "class_group",
        "paper_size",
        "paper_type",
        "pen_type",
        "ink_color",
        "writing_mode",
        "focus_area",
        "content_amount",
        "image_type",
        "lighting",
        "camera_angle",
        "teacher_note",
        "image_path",
        "overall_score",
        "neatness_score",
        "spacing_score",
        "alignment_score",
        "consistency_score",
        "readability_score",
        "letter_formation_score",
        "slant_score",
        "baseline_score",
        "grade",
        "strongest_area",
        "weakest_area",
        "student_summary",
        "teacher_summary",
        "what_you_did_well",
        "what_needs_improvement",
        "how_to_improve",
        "practice_focus",
        "practice_time",
        "practice_exercise",
        "best_paper",
        "encouragement_line",
        "ai_raw_response",
    ]).to_csv(DATA_FILE, index=False)


ensure_results_file()


def save_result(row: dict):
    df_new = pd.DataFrame([row])

    if os.path.exists(DATA_FILE):
        try:
            df_old = pd.read_csv(DATA_FILE)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        except:
            df_all = df_new
    else:
        df_all = df_new

    df_all.to_csv(DATA_FILE, index=False)


def get_all_results() -> pd.DataFrame:
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame()

    try:
        return pd.read_csv(DATA_FILE).fillna("")
    except:
        return pd.DataFrame()


def get_student_df(system_id: str) -> pd.DataFrame:
    df = get_all_results()
    if len(df) == 0:
        return pd.DataFrame()

    return df[df["system_id"].astype(str) == str(system_id)].copy()


def get_previous_attempt(system_id: str):
    df_student = get_student_df(system_id)

    if len(df_student) < 1:
        return None

    df_student = df_student.sort_values("datetime")
    return df_student.iloc[-1]


# =========================================================
# INPUT VALIDATION (IMPROVED)
# =========================================================
def validate_inputs(
    system_id, class_group, paper_size, paper_type, pen_type,
    ink_color, writing_mode, focus_area, content_amount,
    image_type, lighting, camera_angle, uploaded_file
):
    if not str(system_id).isdigit() or len(str(system_id)) != 10:
        return "System ID must be exactly 10 digits."

    required = [
        class_group, paper_size, paper_type, pen_type,
        ink_color, writing_mode, focus_area, content_amount,
        image_type, lighting
    ]

    if "Select..." in required:
        return "Please select all required options."

    if image_type == "Camera Photo" and camera_angle == "Select...":
        return "Please select camera angle."

    if uploaded_file is None:
        return "Please upload an image."

    return None


# =========================================================
# JSON EXTRACTION (VERY IMPORTANT FIX)
# =========================================================
def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    text = text.strip()

    # direct parse
    try:
        return json.loads(text)
    except:
        pass

    # ```json block
    match = re.search(r"```json(.*?)```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except:
            pass

    # fallback find {}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    return None


# =========================================================
# FALLBACK (NO CRASH GUARANTEE)
# =========================================================
def fallback_analysis():
    return {
        "overall_score": 7,
        "scores": {
            "neatness": 7,
            "spacing": 6,
            "alignment": 6,
            "consistency": 7,
            "readability": 7,
            "letter_formation": 6,
            "slant": 6,
            "baseline": 6
        },
        "grade": "Good",
        "strongest_area": "Readability",
        "weakest_area": "Spacing",
        "student_summary": "Your handwriting is readable and shows effort.",
        "teacher_summary": "Moderate handwriting. Improve spacing and alignment.",
        "what_you_did_well": ["Readable writing", "Good effort"],
        "what_needs_improvement": ["Spacing", "Alignment"],
        "how_to_improve": ["Write slower", "Keep spacing consistent"],
        "practice_plan": {
            "focus": "Spacing",
            "daily_time": "10 minutes",
            "exercise": "Practice lines daily",
            "best_paper": "Ruled"
        },
        "encouragement_line": "Keep improving daily!"
    }


# =========================================================
# NORMALIZATION (CRITICAL FIX)
# =========================================================
def normalize_ai_result(data):
    if not isinstance(data, dict):
        return fallback_analysis()

    try:
        scores = data.get("scores", {})

        result = {
            "overall_score": float(data.get("overall_score", 0)),
            "scores": {
                "neatness": float(scores.get("neatness", 0)),
                "spacing": float(scores.get("spacing", 0)),
                "alignment": float(scores.get("alignment", 0)),
                "consistency": float(scores.get("consistency", 0)),
                "readability": float(scores.get("readability", 0)),
                "letter_formation": float(scores.get("letter_formation", 0)),
                "slant": float(scores.get("slant", 0)),
                "baseline": float(scores.get("baseline", 0)),
            },
            "grade": data.get("grade", "Good"),
            "strongest_area": data.get("strongest_area", "Readability"),
            "weakest_area": data.get("weakest_area", "Spacing"),
            "student_summary": data.get("student_summary", ""),
            "teacher_summary": data.get("teacher_summary", ""),
            "what_you_did_well": data.get("what_you_did_well", []),
            "what_needs_improvement": data.get("what_needs_improvement", []),
            "how_to_improve": data.get("how_to_improve", []),
            "practice_plan": data.get("practice_plan", {}),
            "encouragement_line": data.get("encouragement_line", "")
        }

        return result

    except:
        return fallback_analysis()


# =========================================================
# AI ANALYSIS (FIXED + STABLE)
# =========================================================
def analyze_handwriting_with_ai(image: Image.Image, **kwargs):

    prompt = "Analyze handwriting and return JSON only."

    try:
        buffer = BytesIO()
        image.save(buffer, format="PNG")

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                prompt,
                types.Part.from_bytes(
                    data=buffer.getvalue(),
                    mime_type="image/png"
                )
            ]
        )

        response_text = getattr(response, "text", "") or ""

        parsed = extract_json_from_text(response_text)

        if parsed is None:
            fallback = fallback_analysis()
            return {
                "normalized": fallback,
                "raw_text": response_text,
                "used_fallback": True
            }

        normalized = normalize_ai_result(parsed)

        return {
            "normalized": normalized,
            "raw_text": response_text,
            "used_fallback": False
        }

    except Exception as e:
        fallback = fallback_analysis()

        return {
            "normalized": fallback,
            "raw_text": str(e),
            "used_fallback": True
        }


# =========================================================
# GRAPHS (CLEAN + FIXED)
# =========================================================
def show_current_aspect_chart(result):
    scores = result["scores"]

    labels = list(scores.keys())
    values = list(scores.values())

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(labels, values)
    ax.set_ylim(0, 10)
    ax.set_title("Aspect Scores")

    plt.xticks(rotation=25)

    st.pyplot(fig)


def show_overall_progress_graph(system_id):
    df = get_student_df(system_id)

    if len(df) == 0:
        return

    df = df.sort_values("datetime").reset_index(drop=True)

    fig, ax = plt.subplots()

    ax.plot(df.index + 1, df["overall_score"], marker="o")
    ax.set_title("Overall Progress")
    ax.set_ylim(0, 10)

    st.pyplot(fig)

    # =========================================================
# LEADERBOARD HELPERS
# =========================================================
def get_latest_attempts_per_student():
    df = get_all_results()
    if len(df) == 0:
        return pd.DataFrame()

    df = df.copy()
    df["datetime_parsed"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.sort_values("datetime_parsed")

    latest_df = df.groupby("system_id", as_index=False).tail(1)
    return latest_df


def get_top_3_in_class(class_group: str):
    latest_df = get_latest_attempts_per_student()

    if len(latest_df) == 0 or "class_group" not in latest_df.columns:
        return pd.DataFrame()

    class_df = latest_df[latest_df["class_group"].astype(str) == str(class_group)].copy()

    if len(class_df) == 0:
        return pd.DataFrame()

    class_df["overall_score_num"] = pd.to_numeric(
        class_df["overall_score"], errors="coerce"
    ).fillna(0)

    class_df = class_df.sort_values(
        "overall_score_num", ascending=False
    ).head(3).reset_index(drop=True)

    class_df["Rank"] = class_df.index + 1

    return class_df


def get_student_rank_in_class(system_id: str, class_group: str):
    latest_df = get_latest_attempts_per_student()

    if len(latest_df) == 0:
        return None, None

    class_df = latest_df[
        latest_df["class_group"].astype(str) == str(class_group)
    ].copy()

    if len(class_df) == 0:
        return None, None

    class_df["overall_score_num"] = pd.to_numeric(
        class_df["overall_score"], errors="coerce"
    ).fillna(0)

    class_df = class_df.sort_values(
        "overall_score_num", ascending=False
    ).reset_index(drop=True)

    class_df["Rank"] = class_df.index + 1

    me = class_df[
        class_df["system_id"].astype(str) == str(system_id)
    ]

    if len(me) == 0:
        return None, None

    my_rank = int(me.iloc[0]["Rank"])
    top_score = float(class_df.iloc[0]["overall_score_num"])
    my_score = float(me.iloc[0]["overall_score_num"])

    gap = round(top_score - my_score, 2)

    return my_rank, gap


# =========================================================
# LOGIN VIEW (FULL FIXED)
# =========================================================
def show_login_page():

    st.markdown('<div class="main-title">✍️ Handwriting AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">AI-based handwriting analysis for students</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-box">', unsafe_allow_html=True)

    st.markdown("### 🔐 Login")

    role = st.selectbox(
        "Select Role",
        ["Student", "Teacher", "Parent"],
        key="login_role"
    )

    login_id = st.text_input("Login ID", key="login_id_input")

    col1, col2 = st.columns(2)

    # SEND OTP
    with col1:
        if st.button("Send OTP"):
            if not login_id.strip():
                st.error("Enter Login ID")
            else:
                user = get_user(login_id.strip(), role)

                if user is None:
                    st.error("Invalid Login ID")
                else:
                    otp = str(random.randint(100000, 999999))

                    st.session_state.otp = otp
                    st.session_state.otp_login_id = login_id.strip()
                    st.session_state.otp_role = role

                    st.success("OTP sent successfully")
                    st.code(f"Demo OTP: {otp}")

    # OTP INPUT
    otp_input = st.text_input("Enter OTP", type="password")

    # VERIFY OTP
    with col2:
        if st.button("Verify & Login"):

            if not st.session_state.otp:
                st.error("Generate OTP first")

            elif login_id.strip() != str(st.session_state.otp_login_id) or role != st.session_state.otp_role:
                st.error("Mismatch in login details")

            elif otp_input.strip() != str(st.session_state.otp):
                st.error("Incorrect OTP")

            else:
                user = get_user(login_id.strip(), role)

                if user is None:
                    st.error("User not found")
                else:
                    st.session_state.logged_in = True
                    st.session_state.role = role
                    st.session_state.login_id = str(user["login_id"])
                    st.session_state.name = str(user["name"])
                    st.session_state.system_id = str(user["system_id"])
                    st.session_state.class_group = str(user["class_group"])
                    st.session_state.linked_student_id = str(user.get("linked_student_id", ""))

                    # clear OTP
                    st.session_state.otp = None
                    st.session_state.otp_login_id = None
                    st.session_state.otp_role = None

                    st.success(f"Welcome {st.session_state.name}")
                    st.rerun()

    # DEMO USERS
    with st.expander("Show Demo IDs"):
        users_df = load_users()
        st.dataframe(
            users_df[["login_id", "role", "name", "class_group", "linked_student_id"]],
            use_container_width=True
        )

    st.markdown('</div>', unsafe_allow_html=True)


    # =========================================================
# RESULT UI
# =========================================================
def render_analysis_result(result: Dict[str, Any], previous_attempt=None):

    scores = result["scores"]

    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown("## 🤖 AI Handwriting Analysis")

    # OVERALL
    col1, col2, col3 = st.columns(3)

    prev = previous_attempt["overall_score"] if previous_attempt is not None else None

    with col1:
        st.metric("Overall Score", f"{result['overall_score']}/10",
                  delta=safe_delta(result['overall_score'], prev))

    with col2:
        st.success(f"Strongest: {result['strongest_area']}")

    with col3:
        st.error(f"Weakest: {result['weakest_area']}")

    # SUMMARIES
    st.markdown("### 👨‍🎓 Student Summary")
    st.success(result["student_summary"])

    st.markdown("### 📋 Teacher Summary")
    st.info(result["teacher_summary"])

    # SCORES GRID
    st.markdown("### 📊 Scores")

    colA, colB, colC = st.columns(3)

    with colA:
        st.metric("Neatness", scores["neatness"])
        st.metric("Spacing", scores["spacing"])
        st.metric("Alignment", scores["alignment"])

    with colB:
        st.metric("Consistency", scores["consistency"])
        st.metric("Readability", scores["readability"])
        st.metric("Letter Formation", scores["letter_formation"])

    with colC:
        st.metric("Slant", scores["slant"])
        st.metric("Baseline", scores["baseline"])

    # GRAPH
    st.markdown("### 📈 Score Graph")
    show_current_aspect_chart(result)

    # GOOD
    st.markdown("### ✅ What You Did Well")
    for item in result["what_you_did_well"]:
        st.success(item)

    # IMPROVE
    st.markdown("### ⚠️ Needs Improvement")
    for item in result["what_needs_improvement"]:
        st.warning(item)

    # HOW TO IMPROVE
    st.markdown("### 🛠️ Improvement Tips")
    for item in result["how_to_improve"]:
        st.info(item)

    # PRACTICE
    plan = result["practice_plan"]

    st.markdown("### 📚 Practice Plan")
    st.write(f"Focus: {plan.get('focus')}")
    st.write(f"Daily Time: {plan.get('daily_time')}")
    st.write(f"Exercise: {plan.get('exercise')}")
    st.write(f"Best Paper: {plan.get('best_paper')}")

    st.success(result["encouragement_line"])

    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# ATTEMPT HISTORY
# =========================================================
def render_attempt_history(df_me: pd.DataFrame):

    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown("## 🧾 Previous Attempts")

    if len(df_me) == 0:
        st.info("No attempts yet")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    df_me = df_me.sort_values("datetime").reset_index(drop=True)
    df_me["Attempt"] = df_me.index + 1

    st.dataframe(df_me, use_container_width=True)

    attempt_no = st.selectbox("Select Attempt", df_me["Attempt"].tolist())

    row = df_me[df_me["Attempt"] == attempt_no].iloc[0]

    col1, col2 = st.columns([1.2, 1])

    # IMAGE
    with col1:
        img = row.get("image_path", "")
        if img and os.path.exists(img):
            st.image(img, use_container_width=True)
        else:
            st.info("No image")

    # DETAILS
    with col2:
        st.write(f"Date: {row.get('datetime')}")
        st.write(f"Score: {row.get('overall_score')}")
        st.write(f"Grade: {row.get('grade')}")

        pdf = build_single_attempt_pdf(row)

        st.download_button(
            "Download PDF",
            data=pdf,
            file_name=f"attempt_{attempt_no}.pdf",
            mime="application/pdf"
        )

    st.markdown('</div>', unsafe_allow_html=True)


    # =========================================================
# STUDENT VIEW
# =========================================================
def show_student_view():
    system_id = str(st.session_state.system_id)
    class_group = str(st.session_state.class_group)

    st.sidebar.success("Logged in as Student")
    st.sidebar.write(f"Name: {st.session_state.name}")
    st.sidebar.write(f"System ID: {system_id}")
    st.sidebar.write(f"Class: {class_group}")
    if st.sidebar.button("Logout"):
        logout()
        st.rerun()

    st.markdown('<div class="main-title">✍️ Student Handwriting Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Upload handwriting, get AI feedback, track progress, compare class rank, and download reports</div>',
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------
    # STUDENT INFO
    # -----------------------------------------------------
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown("### 👤 Logged-in Student Details")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Student Name", st.session_state.name)
    with c2:
        st.metric("System ID", system_id)
    with c3:
        st.metric("Class Group", class_group)

    st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------
    # LEADERBOARD
    # -----------------------------------------------------
    top3 = get_top_3_in_class(class_group)
    my_rank, gap = get_student_rank_in_class(system_id, class_group)

    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown("### 🏆 Top 3 Students in Your Class")

    if len(top3) > 0:
        for _, row in top3.iterrows():
            sid = str(row.get("system_id", ""))
            display_name = "You" if sid == system_id else mask_id(sid)
            st.markdown(
                f"""
                <div class="result-card result-blue">
                    <b>Rank {int(row["Rank"])}</b> — {display_name} — Overall Score: {row.get("overall_score", 0)}/10
                </div>
                """,
                unsafe_allow_html=True,
            )

        if my_rank is not None:
            st.info(f"Your current class rank is {my_rank}. Gap from rank 1: {gap}")
    else:
        st.info("Leaderboard will appear after students in your class have attempts saved.")

    st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------
    # WRITING DETAILS
    # -----------------------------------------------------
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown("### 📘 Writing Details")

    col3, col4, col5 = st.columns(3)
    with col3:
        paper_size = st.selectbox(
            "Paper Size",
            ["Select...", "A5", "A4", "A3", "A2", "Chart Paper", "Notebook Size"],
            key="s_paper_size"
        )
    with col4:
        paper_type = st.selectbox(
            "Paper Type",
            ["Select...", "Ruled", "Unruled", "Graph"],
            key="s_paper_type"
        )
    with col5:
        pen_type = st.selectbox(
            "Pen Type",
            ["Select...", "Ball Pen", "Gel Pen", "Pencil"],
            key="s_pen_type"
        )

    col6, col7 = st.columns(2)
    with col6:
        ink_color = st.selectbox(
            "Ink Color",
            ["Select...", "Blue", "Black", "Red"],
            key="s_ink_color"
        )
    with col7:
        writing_mode = st.selectbox(
            "Writing Mode",
            ["Select...", "Practice", "Homework", "Classwork", "Exam"],
            key="s_writing_mode"
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------
    # ANALYSIS SETTINGS
    # -----------------------------------------------------
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown("### 📝 Analysis Settings")

    col8, col9 = st.columns(2)
    with col8:
        focus_area = st.selectbox(
            "Main Focus for Analysis",
            ["Select...", "Overall Quality", "Neatness", "Spacing", "Alignment", "Letter Formation", "Readability"],
            key="s_focus_area"
        )
    with col9:
        content_amount = st.selectbox(
            "Amount of Writing",
            ["Select...", "1–2 Lines", "Short Paragraph", "Half Page", "Full Page"],
            key="s_content_amount"
        )

    teacher_note = st.text_area(
        "Optional Note",
        placeholder="Example: This is today's classwork page. Please focus on spacing and neatness.",
        key="s_teacher_note"
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------
    # IMAGE DETAILS
    # -----------------------------------------------------
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown("### 📷 Image Details")

    col10, col11 = st.columns(2)
    with col10:
        image_type = st.selectbox(
            "Image Type",
            ["Select...", "Camera Photo", "Scanned Image"],
            key="s_image_type"
        )
    with col11:
        lighting = st.selectbox(
            "Lighting Condition",
            ["Select...", "Good", "Average", "Poor"],
            key="s_lighting"
        )

    camera_angle = "Not Required"
    if image_type == "Camera Photo":
        camera_angle = st.selectbox(
            "Camera Angle",
            ["Select...", "Straight", "Slight Tilt", "High Tilt"],
            key="s_camera_angle"
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------
    # UPLOAD
    # -----------------------------------------------------
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload Handwriting Image")

    uploaded_file = st.file_uploader(
        "Choose a handwriting image",
        type=["jpg", "jpeg", "png", "webp"],
        key="student_upload"
    )

    st.caption("Tip: Clear, non-blurry images work best.")

    if uploaded_file is not None:
        try:
            preview_image = Image.open(uploaded_file).convert("RGB")
            st.image(preview_image, caption="Preview", use_container_width=True)
        except Exception:
            st.warning("Could not preview image.")

    st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------
    # ANALYZE
    # -----------------------------------------------------
    if st.button("Analyze Handwriting", key="student_analyze_btn"):
        error = validate_inputs(
            system_id, class_group, paper_size, paper_type, pen_type,
            ink_color, writing_mode, focus_area, content_amount,
            image_type, lighting, camera_angle, uploaded_file
        )

        if error:
            st.error(error)
        else:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                previous_attempt = get_previous_attempt(system_id)
                image_path = save_uploaded_image(uploaded_file, system_id)

                with st.spinner("Analyzing handwriting with AI..."):
                    ai_bundle = analyze_handwriting_with_ai(
                        image=image,
                        class_group=class_group,
                        paper_size=paper_size,
                        paper_type=paper_type,
                        pen_type=pen_type,
                        ink_color=ink_color,
                        writing_mode=writing_mode,
                        focus_area=focus_area,
                        content_amount=content_amount,
                        image_type=image_type,
                        lighting=lighting,
                        camera_angle=camera_angle,
                        teacher_note=teacher_note,
                    )

                result = ai_bundle["normalized"]
                raw_response = ai_bundle["raw_text"]
                used_fallback = ai_bundle["used_fallback"]

                if used_fallback:
                    st.warning("AI output was not perfectly structured, so a safe fallback result was used for display.")

                save_result({
                    "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "viewer_role": st.session_state.role,
                    "login_id": st.session_state.login_id,
                    "student_name": st.session_state.name,
                    "system_id": system_id,
                    "class_group": class_group,
                    "paper_size": paper_size,
                    "paper_type": paper_type,
                    "pen_type": pen_type,
                    "ink_color": ink_color,
                    "writing_mode": writing_mode,
                    "focus_area": focus_area,
                    "content_amount": content_amount,
                    "image_type": image_type,
                    "lighting": lighting,
                    "camera_angle": camera_angle,
                    "teacher_note": teacher_note,
                    "image_path": image_path,
                    "overall_score": result["overall_score"],
                    "neatness_score": result["scores"]["neatness"],
                    "spacing_score": result["scores"]["spacing"],
                    "alignment_score": result["scores"]["alignment"],
                    "consistency_score": result["scores"]["consistency"],
                    "readability_score": result["scores"]["readability"],
                    "letter_formation_score": result["scores"]["letter_formation"],
                    "slant_score": result["scores"]["slant"],
                    "baseline_score": result["scores"]["baseline"],
                    "grade": result["grade"],
                    "strongest_area": result["strongest_area"],
                    "weakest_area": result["weakest_area"],
                    "student_summary": result["student_summary"],
                    "teacher_summary": result["teacher_summary"],
                    "what_you_did_well": json.dumps(result["what_you_did_well"], ensure_ascii=False),
                    "what_needs_improvement": json.dumps(result["what_needs_improvement"], ensure_ascii=False),
                    "how_to_improve": json.dumps(result["how_to_improve"], ensure_ascii=False),
                    "practice_focus": result["practice_plan"]["focus"],
                    "practice_time": result["practice_plan"]["daily_time"],
                    "practice_exercise": result["practice_plan"]["exercise"],
                    "best_paper": result["practice_plan"]["best_paper"],
                    "encouragement_line": result["encouragement_line"],
                    "ai_raw_response": raw_response,
                })

                st.session_state.last_analysis_result = result
                render_analysis_result(result, previous_attempt=previous_attempt)

                # PROGRESS
                st.markdown('<div class="section-box">', unsafe_allow_html=True)
                st.markdown("### 📈 Progress Tracking")

                st.markdown("#### Overall Progress")
                show_overall_progress_graph(system_id)

                st.markdown("#### Detailed Progress by Aspect")
                show_aspect_progress_graph(system_id)

                st.markdown('</div>', unsafe_allow_html=True)

                # 3-MONTH REPORT
                st.markdown('<div class="section-box">', unsafe_allow_html=True)
                st.markdown("### 📄 Download 3-Month Progress PDF")

                df_student_report = get_student_df(system_id)
                earlier_row, latest_row = get_three_month_comparison_rows(df_student_report)

                if earlier_row is not None and latest_row is not None:
                    pdf_buffer = build_three_month_pdf(system_id, earlier_row, latest_row)
                    st.download_button(
                        "Download 3-Month PDF Report",
                        data=pdf_buffer,
                        file_name=f"{system_id}_3_month_progress_report.pdf",
                        mime="application/pdf"
                    )
                    st.info(
                        f"Report compares: {safe_text(earlier_row.get('datetime'))} vs {safe_text(latest_row.get('datetime'))}"
                    )
                else:
                    st.info("Not enough data yet to generate a 3-month comparison report.")

                st.markdown('</div>', unsafe_allow_html=True)

                with st.expander("Show Full AI Raw Response"):
                    st.write(raw_response)

            except Exception as e:
                st.error(f"Error while analyzing image: {e}")

    # -----------------------------------------------------
    # HISTORY
    # -----------------------------------------------------
    df_me = get_student_df(system_id)
    render_attempt_history(df_me)

    st.markdown(
        '<div class="sub-title">Built as a creative school project for handwriting analysis and improvement</div>',
        unsafe_allow_html=True,
    )


# =========================================================
# PARENT VIEW
# =========================================================
def show_parent_view():
    linked_student_id = get_linked_student_id_for_current_user()

    st.sidebar.success("Logged in as Parent")
    st.sidebar.write(f"Name: {st.session_state.name}")
    st.sidebar.write(f"Login ID: {st.session_state.login_id}")
    st.sidebar.write(f"Linked Student ID: {linked_student_id}")
    if st.sidebar.button("Logout", key="parent_logout_btn"):
        logout()
        st.rerun()

    st.markdown('<div class="main-title">👨‍👩‍👧 Parent Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">See your child’s handwriting progress, reports, scores, and practice plan</div>',
        unsafe_allow_html=True,
    )

    if not linked_student_id:
        st.warning("No student is linked to this parent account.")
        return

    df_student = get_student_df(linked_student_id)
    if len(df_student) == 0:
        st.info("No handwriting attempts found yet for the linked student.")
        return

    df_student = df_student.sort_values("datetime").reset_index(drop=True)
    latest = df_student.iloc[-1]

    student_name = safe_text(latest.get("student_name"), "Student")
    class_group = safe_text(latest.get("class_group"), "N/A")

    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown("### Child Overview")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Student Name", student_name)
    with c2:
        st.metric("Student ID", linked_student_id)
    with c3:
        st.metric("Class Group", class_group)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown("### 📌 Latest Handwriting Performance")

    d1, d2, d3 = st.columns(3)
    with d1:
        st.metric("Latest Overall", f'{latest.get("overall_score", 0)}/10')
    with d2:
        st.metric("Grade", latest.get("grade", ""))
    with d3:
        st.metric("Total Attempts", len(df_student))

    st.markdown(
        f'<div class="result-card result-green"><b>Student Summary:</b> {latest.get("student_summary", "")}</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="result-card result-purple"><b>Teacher Summary:</b> {latest.get("teacher_summary", "")}</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="result-card result-blue"><b>Practice Focus:</b> {latest.get("practice_focus", "")}</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="result-card result-orange"><b>Daily Time:</b> {latest.get("practice_time", "")}</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="result-card result-purple"><b>Exercise:</b> {latest.get("practice_exercise", "")}</div>',
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown("### 📈 Child Progress")

    show_overall_progress_graph(linked_student_id)
    show_aspect_progress_graph(linked_student_id)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown("### 📄 Download Reports")

    earlier_row, latest_row = get_three_month_comparison_rows(df_student)
    if earlier_row is not None and latest_row is not None:
        parent_pdf = build_three_month_pdf(linked_student_id, earlier_row, latest_row)
        st.download_button(
            "Download Child 3-Month PDF Report",
            data=parent_pdf,
            file_name=f"{linked_student_id}_parent_view_3_month_report.pdf",
            mime="application/pdf"
        )

    latest_pdf = build_single_attempt_pdf(latest)
    st.download_button(
        "Download Latest Attempt PDF",
        data=latest_pdf,
        file_name=f"{linked_student_id}_latest_attempt_report.pdf",
        mime="application/pdf"
    )

    st.markdown('</div>', unsafe_allow_html=True)

    render_attempt_history(df_student)

    # =========================================================
# TEACHER VIEW
# =========================================================
def show_teacher_view():
    st.sidebar.success("Logged in as Teacher")
    st.sidebar.write(f"Name: {st.session_state.name}")
    st.sidebar.write(f"Login ID: {st.session_state.login_id}")
    st.sidebar.write(f"Class Group: {st.session_state.class_group}")

    if st.sidebar.button("Logout", key="teacher_logout_btn"):
        logout()
        st.rerun()

    st.markdown('<div class="main-title">🧑‍🏫 Teacher Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Track students, analyze handwriting, monitor progress, and download reports</div>',
        unsafe_allow_html=True,
    )

    df = get_all_results()

    if len(df) == 0:
        st.warning("No data available yet.")
        return

    # =====================================================
    # FILTER ONLY TEACHER CLASS (IMPORTANT FIX)
    # =====================================================
    teacher_class = str(st.session_state.class_group)
    df = df[df["class_group"].astype(str) == teacher_class]

    # =====================================================
    # OVERVIEW
    # =====================================================
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown("### 📊 Class Overview")

    total_attempts = len(df)
    total_students = df["system_id"].astype(str).nunique()

    avg_score = round(
        pd.to_numeric(df["overall_score"], errors="coerce").fillna(0).mean(),
        2
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Students", total_students)
    with c2:
        st.metric("Total Attempts", total_attempts)
    with c3:
        st.metric("Average Score", f"{avg_score}/10")

    st.markdown('</div>', unsafe_allow_html=True)

    # =====================================================
    # TEACHER ANALYSIS TOOL
    # =====================================================
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown("### 🧪 Teacher Analysis Tool")

    users = load_users()
    students = users[users["role"] == "Student"]

    # FILTER BY TEACHER CLASS
    students = students[students["class_group"] == teacher_class]

    if len(students) == 0:
        st.info("No students found for your class.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    student_map = {
        f'{row["name"]} ({row["system_id"]})': row["system_id"]
        for _, row in students.iterrows()
    }

    selected_label = st.selectbox("Select Student", list(student_map.keys()))
    selected_student_id = student_map[selected_label]

    selected_row = students[students["system_id"] == selected_student_id].iloc[0]
    selected_name = selected_row["name"]
    selected_class = selected_row["class_group"]

    # INPUTS
    col1, col2, col3 = st.columns(3)
    with col1:
        paper_size = st.selectbox("Paper Size", ["Select...", "A4", "A5"], key="t1")
    with col2:
        paper_type = st.selectbox("Paper Type", ["Select...", "Ruled", "Unruled"], key="t2")
    with col3:
        pen_type = st.selectbox("Pen Type", ["Select...", "Ball", "Gel"], key="t3")

    col4, col5 = st.columns(2)
    with col4:
        ink_color = st.selectbox("Ink Color", ["Select...", "Blue", "Black"], key="t4")
    with col5:
        writing_mode = st.selectbox("Mode", ["Select...", "Practice", "Exam"], key="t5")

    focus_area = st.selectbox("Focus", ["Select...", "Neatness", "Spacing"], key="t6")
    content_amount = st.selectbox("Amount", ["Select...", "Short", "Full"], key="t7")

    image_type = st.selectbox("Image Type", ["Select...", "Camera Photo", "Scan"], key="t8")
    lighting = st.selectbox("Lighting", ["Select...", "Good", "Poor"], key="t9")

    camera_angle = "Not Required"
    if image_type == "Camera Photo":
        camera_angle = st.selectbox("Angle", ["Select...", "Straight"], key="t10")

    teacher_note = st.text_area("Teacher Note", key="t11")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"], key="t12")

    if uploaded_file:
        st.image(uploaded_file, use_container_width=True)

    if st.button("Analyze Student Handwriting", key="teacher_analyze"):
        error = validate_inputs(
            selected_student_id, selected_class,
            paper_size, paper_type, pen_type,
            ink_color, writing_mode, focus_area,
            content_amount, image_type, lighting,
            camera_angle, uploaded_file
        )

        if error:
            st.error(error)
        else:
            image = Image.open(uploaded_file).convert("RGB")
            prev = get_previous_attempt(selected_student_id)
            path = save_uploaded_image(uploaded_file, selected_student_id)

            ai = analyze_handwriting_with_ai(image=image)

            result = ai["normalized"]

            save_result({
                "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "viewer_role": st.session_state.role,
                "login_id": st.session_state.login_id,
                "student_name": selected_name,
                "system_id": selected_student_id,
                "class_group": selected_class,
                "paper_size": paper_size,
                "paper_type": paper_type,
                "pen_type": pen_type,
                "ink_color": ink_color,
                "writing_mode": writing_mode,
                "focus_area": focus_area,
                "content_amount": content_amount,
                "image_type": image_type,
                "lighting": lighting,
                "camera_angle": camera_angle,
                "teacher_note": teacher_note,
                "image_path": path,
                "overall_score": result["overall_score"],
                "neatness_score": result["scores"]["neatness"],
                "spacing_score": result["scores"]["spacing"],
                "alignment_score": result["scores"]["alignment"],
                "consistency_score": result["scores"]["consistency"],
                "readability_score": result["scores"]["readability"],
                "letter_formation_score": result["scores"]["letter_formation"],
                "slant_score": result["scores"]["slant"],
                "baseline_score": result["scores"]["baseline"],
                "grade": result["grade"],
                "strongest_area": result["strongest_area"],
                "weakest_area": result["weakest_area"],
                "student_summary": result["student_summary"],
                "teacher_summary": result["teacher_summary"],
                "what_you_did_well": json.dumps(result["what_you_did_well"]),
                "what_needs_improvement": json.dumps(result["what_needs_improvement"]),
                "how_to_improve": json.dumps(result["how_to_improve"]),
                "practice_focus": result["practice_plan"]["focus"],
                "practice_time": result["practice_plan"]["daily_time"],
                "practice_exercise": result["practice_plan"]["exercise"],
                "best_paper": result["practice_plan"]["best_paper"],
                "encouragement_line": result["encouragement_line"],
                "ai_raw_response": ai["raw_text"],
            })

            render_analysis_result(result, previous_attempt=prev)

    st.markdown('</div>', unsafe_allow_html=True)

    # =====================================================
    # FULL DATA
    # =====================================================
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.markdown("### 📚 Full Data Table")

    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "data.csv")

    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# MAIN
# =========================================================
if not st.session_state.logged_in:
    show_login_page()
else:
    if st.session_state.role == "Student":
        show_student_view()
    elif st.session_state.role == "Teacher":
        show_teacher_view()
    elif st.session_state.role == "Parent":
        show_parent_view()
    else:
        logout()
        st.rerun()
