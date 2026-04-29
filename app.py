import streamlit as st
import os
import asyncio
import sqlite3
import random
import tempfile
import pandas as pd
import pytz
from datetime import datetime
from audio_recorder_streamlit import audio_recorder
from langchain_core.messages import HumanMessage
from backend.main import create_pdf_buffer
from backend.agents.graph import coaching_agent_app
from backend.agents.specialized import QUOTES

# ─── Istanbul Timezone ──────────────────────────────────────────────────────────
istanbul = pytz.timezone("Europe/Istanbul")

def now_istanbul():
    return datetime.now(istanbul)

# ─── STT via Groq Whisper ──────────────────────────────────────────────────────
def transcribe_audio_with_groq(audio_bytes: bytes) -> str:
    """Send audio bytes to Groq Whisper API for speech-to-text transcription."""
    try:
        import tempfile
        from groq import Groq
        groq_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
        if not groq_key:
            return ""
        client = Groq(api_key=groq_key)
        # Write audio bytes to a temp file (Groq requires a file-like object)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        with open(tmp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file,
                language="tr"
            )
        return transcription.text.strip()
    except Exception as e:
        st.warning(f"Ses tanıma hatası: {e}")
        return ""

# ─── Async Agent Invocation ─────────────────────────────────────────────────────
def run_agent(state_update: dict, config: dict) -> dict:
    """Run the LangGraph agent, handling existing event loops safely."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already inside a running loop (e.g., Streamlit Cloud) — use a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coaching_agent_app.ainvoke(state_update, config))
                return future.result()
        else:
            return loop.run_until_complete(coaching_agent_app.ainvoke(state_update, config))
    except RuntimeError:
        return asyncio.run(coaching_agent_app.ainvoke(state_update, config))

# --- PAGE CONFIG ---
st.set_page_config(page_title="IREM AI - Eğitim Koçu", layout="wide", page_icon="💠")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    * { font-family: 'Outfit', sans-serif; }
    
    .stApp {
        background: #0F172A;
        color: #F8FAFC;
    }
    
    [data-testid="stSidebar"] {
        background: #111827 !important;
        border-right: 1px solid rgba(124, 58, 237, 0.2);
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(124, 58, 237, 0.3);
        border-radius: 20px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .chat-bubble {
        padding: 15px 20px;
        border-radius: 20px;
        margin-bottom: 15px;
        max-width: 85%;
    }
    
    .user-msg {
        background: linear-gradient(135deg, #7C3AED 0%, #6D28D9 100%);
        color: white;
        margin-left: auto;
    }
    
    .assistant-msg {
        background: #1E293B;
        color: #F1F5F9;
        margin-right: auto;
        border-left: 4px solid #7C3AED;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #7C3AED 0%, #EC4899 100%);
        color: white;
        border-radius: 12px;
        font-weight: 600;
    }

    .workspace-box {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(124, 58, 237, 0.4);
        border-radius: 16px;
        padding: 20px;
        min-height: 380px;
        max-height: 400px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# --- STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_name" not in st.session_state:
    st.session_state.user_name = "Öğrenci"
if "grade" not in st.session_state:
    st.session_state.grade = "12. Sınıf"
if "mood" not in st.session_state:
    st.session_state.mood = "Pozitif"
if "energy" not in st.session_state:
    st.session_state.energy = 85
if "current_panel" not in st.session_state:
    st.session_state.current_panel = "dashboard"
if "workspace_content" not in st.session_state:
    st.session_state.workspace_content = ""
if "goals_list" not in st.session_state:
    st.session_state.goals_list = []
if "schedule_list" not in st.session_state:
    st.session_state.schedule_list = []
if "audio_processed" not in st.session_state:
    st.session_state.audio_processed = False

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h1 style='text-align:center;'>💠 IREM AI</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.session_state.user_name = st.text_input("Adın Soyadın", value=st.session_state.user_name)
    st.session_state.grade = st.selectbox("Sınıfın", ["1. Sınıf", "2. Sınıf", "3. Sınıf", "4. Sınıf", "5. Sınıf", "6. Sınıf", "7. Sınıf", "8. Sınıf", "9. Sınıf", "10. Sınıf", "11. Sınıf", "12. Sınıf", "Mezun", "Üniversite"], index=8)
    
    if st.sidebar.button("📊 Dashboard", use_container_width=True, key="nav_dash"):
        st.session_state.current_panel = "dashboard"
        st.rerun()
    if st.sidebar.button("🎯 Hedeflerim", use_container_width=True, key="nav_goals"):
        st.session_state.current_panel = "goals"
        st.rerun()
    if st.sidebar.button("📅 Çalışma Takvimi", use_container_width=True, key="nav_cal"):
        st.session_state.current_panel = "calendar"
        st.rerun()
    
    st.markdown("---")
    st.progress(st.session_state.energy / 100, text=f"Enerji: %{st.session_state.energy}")

# --- MAIN DASHBOARD ---
if st.session_state.current_panel == "dashboard":
    now = now_istanbul()

    # ─── Dashboard reminder check ────────────────────────────────────────────────
    now_ts = now_istanbul()
    for _s in st.session_state.schedule_list:
        try:
            _sd = datetime.strptime(_s["day"], "%Y-%m-%d").date()
            if _sd == now_ts.date():
                _st = datetime.strptime(_s["start_time"], "%H:%M:%S").time()
                _diff = (datetime.combine(_sd, _st) - now_ts.replace(tzinfo=None)).total_seconds()
                if 0 <= _diff <= 1800:
                    st.toast(f"🔔 Yaklaşan Oturum: {_s['task']} — {_s['start_time']}", icon="📅")
                elif -60 <= _diff < 0:
                    st.warning(f"⏰ **{_s['task']}** çalışma oturumun şu an başlamalı!")
        except Exception:
            pass

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.markdown(f'<div class="metric-card"><div style="color:#94A3B8;">ODAK</div><div style="font-size:1.8rem; font-weight:700; color:#7C3AED;">%{st.session_state.energy}</div></div>', unsafe_allow_html=True)
    with m2: st.markdown(f'<div class="metric-card"><div style="color:#94A3B8;">MOD</div><div style="font-size:1.8rem; font-weight:700; color:#EC4899;">{st.session_state.mood}</div></div>', unsafe_allow_html=True)
    with m3: st.markdown(f'<div class="metric-card"><div style="color:#94A3B8;">GÜN</div><div style="font-size:1.8rem; font-weight:700; color:#3B82F6;">{now.strftime("%d %b")}</div></div>', unsafe_allow_html=True)
    with m4: st.markdown(f'<div class="metric-card"><div style="color:#94A3B8;">SAAT</div><div style="font-size:1.8rem; font-weight:700; color:#10B981;">{now.strftime("%H:%M")}</div></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Quote of the day
    all_quotes = QUOTES.get("Motivation", [{"text": "Başarı, çabanın sonucudur.", "author": "İrem AI", "emoji": "✨"}])
    quote = random.choice(all_quotes)
    st.markdown(f"""
        <div style="background: rgba(124, 58, 237, 0.1); padding: 25px; border-radius: 20px; border: 1px solid rgba(124, 58, 237, 0.3); text-align: center;">
            <div style="font-size:11px; color:#F472B6; font-weight:800; letter-spacing:2px; margin-bottom:5px;">✨ GÜNÜN SÖZÜ</div>
            <div style="font-size:1.1rem; font-weight:600; color:white;">"{quote['text']}"</div>
            <div style="font-size:0.9rem; color:#F472B6; margin-top:5px;">— {quote['author']} {quote['emoji']}</div>
        </div>
    """, unsafe_allow_html=True)

    col_chat, col_doc = st.columns([1.1, 1.4], gap="large")
    
    with col_chat:
        chat_sub = st.container(height=450, border=False)
        with chat_sub:
            if not st.session_state.messages:
                st.markdown(f'<div class="chat-bubble assistant-msg">Merhaba {st.session_state.user_name}! Bugün sana nasıl yardımcı olabilirim? 💠</div>', unsafe_allow_html=True)
            for m in st.session_state.messages:
                cls = "user-msg" if m["role"] == "user" else "assistant-msg"
                st.markdown(f'<div class="chat-bubble {cls}">{m["content"]}</div>', unsafe_allow_html=True)
        
        # ─── Audio Input ────────────────────────────────────────────────────────
        audio_bytes = audio_recorder(text="🎙️", icon_size="2x", key="recorder_dash")
        prompt = st.chat_input("Mesajınızı yazın...")

        # STT: Process recorded audio via Gemini
        if audio_bytes and len(audio_bytes) > 1000 and not st.session_state.audio_processed:
            st.session_state.audio_processed = True
            with st.spinner("🎙️ Sesiniz metne çevriliyor..."):
                transcribed = transcribe_audio_with_groq(audio_bytes)
            if transcribed:
                st.success(f"Anlaşılan: _{transcribed}_")
                prompt = transcribed
            else:
                st.warning("Ses anlaşılamadı, lütfen tekrar deneyin.")
        elif not audio_bytes:
            st.session_state.audio_processed = False  # Reset for next recording

        # ─── Process Prompt ─────────────────────────────────────────────────────
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            config = {"configurable": {"thread_id": "st-session-1"}}
            state_update = {
                "messages": [HumanMessage(content=prompt)],
                "user_name": st.session_state.user_name,
                "language": "tr"
            }
            
            with st.spinner("💠 Düşünüyorum..."):
                try:
                    response_state = run_agent(state_update, config)
                    last_msg = response_state["messages"][-1].content
                    st.session_state.energy = response_state.get("energy_score", st.session_state.energy)
                    st.session_state.messages.append({"role": "assistant", "content": last_msg})
                    # ✅ Always update workspace with every AI response
                    st.session_state.workspace_content = last_msg
                except Exception as e:
                    err_msg = f"⚠️ Hata oluştu: {e}"
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})
                    st.session_state.workspace_content = err_msg
            
            st.rerun()

    # ─── Workspace Panel ────────────────────────────────────────────────────────
    with col_doc:
        st.markdown("### 📄 Çalışma Alanı")
        if st.session_state.workspace_content:
            content = st.session_state.workspace_content
            with st.container(height=400, border=True):
                st.markdown(content)
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    "📥 Notu İndir (.txt)",
                    data=content.encode("utf-8"),
                    file_name="irem_not.txt",
                    mime="text/plain; charset=utf-8",
                    use_container_width=True
                )
            with col_dl2:
                if st.button("🗑️ Temizle", use_container_width=True):
                    st.session_state.workspace_content = ""
                    st.rerun()
        else:
            st.markdown("""
            <div style="background: rgba(30,41,59,0.5); border: 1px dashed rgba(124,58,237,0.3); border-radius:16px; padding:40px; text-align:center; color:#64748B; min-height:380px; display:flex; flex-direction:column; justify-content:center;">
                <div style="font-size:2rem;">📝</div>
                <div style="margin-top:10px;">Yapay zekaya bir şey sor veya bir içerik iste.<br>Yanıt burada görünecek.</div>
            </div>
            """, unsafe_allow_html=True)

elif st.session_state.current_panel == "goals":
    st.markdown("## 🎯 Hedeflerim")
    col_f, col_v = st.columns([1, 1.5])
    
    with col_f:
        with st.form("goal_form_modern"):
            g_title = st.text_input("Hedef Adı")
            g_desc = st.text_area("Açıklama")
            g_start = st.date_input("Başlangıç Tarihi", now_istanbul())
            g_end = st.date_input("Bitiş Tarihi", now_istanbul())
            if st.form_submit_button("Hedef Ekle"):
                new_goal = {"title": g_title, "desc": g_desc, "start": str(g_start), "end": str(g_end), "status": "Devam Ediyor"}
                st.session_state.goals_list.append(new_goal)
                conn = sqlite3.connect('goals.db')
                c = conn.cursor()
                c.execute("CREATE TABLE IF NOT EXISTS goals_v2 (title TEXT, description TEXT, start_date TEXT, end_date TEXT, status TEXT)")
                c.execute("INSERT INTO goals_v2 VALUES (?, ?, ?, ?, ?)", (g_title, g_desc, str(g_start), str(g_end), "Devam Ediyor"))
                conn.commit(); conn.close()
                st.success("Hedef başarıyla eklendi!")

    with col_v:
        st.markdown("### 📅 Aylık Hedef Takvimi")
        if st.session_state.goals_list:
            import plotly.express as px
            df_goals = pd.DataFrame(st.session_state.goals_list)
            df_goals['start'] = pd.to_datetime(df_goals['start'])
            df_goals['end'] = pd.to_datetime(df_goals['end'])
            fig = px.timeline(df_goals, x_start="start", x_end="end", y="title", color="status",
                             title="Hedef Zaman Çizelgesi", template="plotly_dark")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Henüz bir hedef eklemediniz.")
    
    st.markdown("---")
    st.markdown("### Tüm Hedefler")
    if st.session_state.goals_list:
        st.table(st.session_state.goals_list)

elif st.session_state.current_panel == "calendar":
    st.markdown("## 📅 Çalışma Takvimi")
    now = now_istanbul()
    
    # Reminder logic
    for session in st.session_state.schedule_list:
        try:
            s_date = datetime.strptime(session["day"], "%Y-%m-%d").date()
            if s_date == now.date():
                s_time = datetime.strptime(session["start_time"], "%H:%M:%S").time()
                diff = (datetime.combine(now.date(), s_time) - now.replace(tzinfo=None)).total_seconds()
                if 0 <= diff <= 1800:
                    st.toast(f"🔔 Yaklaşan Oturum: {session['task']} (Saat {session['start_time']})", icon="📅")
        except Exception:
            pass

    col_f, col_v = st.columns([1, 2])
    
    with col_f:
        with st.form("cal_form_modern"):
            c_task = st.text_input("Görev/Ders")
            c_day = st.date_input("Gün", now)
            c_start_t = st.time_input("Başlangıç Saati", value=now.time(), step=60)
            c_end_t = st.time_input("Bitiş Saati", value=now.time(), step=60)
            if st.form_submit_button("Takvime Ekle"):
                new_session = {
                    "task": c_task,
                    "day": str(c_day),
                    "start_time": str(c_start_t),
                    "end_time": str(c_end_t),
                    "start": f"{c_day} {c_start_t}",
                    "end": f"{c_day} {c_end_t}"
                }
                st.session_state.schedule_list.append(new_session)
                st.success(f"{c_task} kaydedildi!")

    with col_v:
        st.markdown("### 🗓️ Haftalık Görünüm")
        if st.session_state.schedule_list:
            import plotly.express as px
            df_sched = pd.DataFrame(st.session_state.schedule_list)
            df_sched['start'] = pd.to_datetime(df_sched['start'])
            df_sched['end'] = pd.to_datetime(df_sched['end'])
            fig = px.timeline(df_sched, x_start="start", x_end="end", y="task", color="task",
                             title="Haftalık Çalışma Planı", template="plotly_dark")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Henüz bir çalışma oturumu eklemediniz.")
