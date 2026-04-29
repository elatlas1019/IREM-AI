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
from streamlit_autorefresh import st_autorefresh

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
if "shown_reminders" not in st.session_state:
    st.session_state.shown_reminders = set()
if "cal_start_time" not in st.session_state:
    st.session_state.cal_start_time = datetime.now(pytz.timezone("Europe/Istanbul")).replace(second=0, microsecond=0).time()
if "cal_end_time" not in st.session_state:
    st.session_state.cal_end_time = datetime.now(pytz.timezone("Europe/Istanbul")).replace(second=0, microsecond=0).time()

# ─── Auto-refresh every 60 seconds for reminder checks ──────────────────────────
st_autorefresh(interval=60_000, key="reminder_refresh")

# ─── Global reminder check (every page load / auto-refresh) ─────────────────────
_now = datetime.now(pytz.timezone("Europe/Istanbul"))
_active_task = None

for _s in st.session_state.schedule_list:
    try:
        _start_dt = datetime.strptime(f"{_s['day']} {_s['start_time']}", "%Y-%m-%d %H:%M:%S")
        _end_dt = datetime.strptime(f"{_s['day']} {_s['end_time']}", "%Y-%m-%d %H:%M:%S")
        _now_dt = _now.replace(tzinfo=None)
        
        if _start_dt <= _now_dt <= _end_dt:
            _active_task = _s
            break  # Show only one active task
    except Exception:
        pass

if _active_task:
    # Render animated banner
    _task_name = _active_task['task']
    _start_str = _active_task['start_time'][:5]
    _end_str = _active_task['end_time'][:5]
    
    st.markdown(f"""
        <style>
        @keyframes pulse-border {{
            0% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }}
            70% {{ box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }}
        }}
        .active-reminder-banner {{
            background: linear-gradient(90deg, rgba(239,68,68,0.1) 0%, rgba(185,28,28,0.2) 100%);
            border: 2px solid #EF4444;
            border-radius: 12px;
            padding: 15px 20px;
            margin-bottom: 20px;
            color: #FCA5A5;
            font-weight: 600;
            display: flex;
            align-items: center;
            justify-content: center;
            animation: pulse-border 2s infinite;
        }}
        </style>
        <div class="active-reminder-banner">
            ⏰ Çalışma zamanı! Görev: {_task_name} | Başlangıç: {_start_str} &rarr; Bitiş: {_end_str}
        </div>
    """, unsafe_allow_html=True)
    
    # Toast trigger (only once)
    _task_id = f"{_task_name}_{_active_task['day']}_{_active_task['start_time']}"
    if _task_id not in st.session_state.shown_reminders:
        st.toast(f"⏰ Çalışma zamanı! Görev: {_task_name}", icon="⏰")
        st.session_state.shown_reminders.add(_task_id)


# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
        <div style="text-align:center; padding: 10px 0 10px 0;">
            <div style="font-size: 2.2rem; font-weight: 800; background: linear-gradient(90deg, #C084FC, #EC4899, #3B82F6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: 1px;">
                ✨ IREM AI
            </div>
            <div style="font-size: 0.85rem; color: #94A3B8; font-style: italic; margin-top: 5px;">
                Kişisel Öğrenme Koçun 🚀
            </div>
            <hr style="border-color: rgba(124, 58, 237, 0.2); margin-top: 15px; margin-bottom: 5px;">
        </div>
    """, unsafe_allow_html=True)
    
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

    # ─── Dashboard reminder check (1-minute precision) ────────────────────────────────
    now_ts = now_istanbul()
    for _s in st.session_state.schedule_list:
        try:
            _sd = datetime.strptime(_s["day"], "%Y-%m-%d").date()
            if _sd == now_ts.date():
                _st_time = datetime.strptime(_s["start_time"], "%H:%M:%S").time()
                _diff = (datetime.combine(_sd, _st_time) - now_ts.replace(tzinfo=None)).total_seconds()
                if abs(_diff) <= 60:  # Within 1 minute of start time
                    st.toast(f"⏰ Çalışma zamanı! Görev: {_s['task']}", icon="⏰")
                    st.warning(f"⏰ Çalışma zamanı! **Görev: {_s['task']}** — {_s['start_time']}")
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
                    # Request 1: Workspace content via run_agent
                    response_state = run_agent(state_update, config)
                    workspace_msg = response_state["messages"][-1].content
                    st.session_state.energy = response_state.get("energy_score", st.session_state.energy)
                    st.session_state.workspace_content = workspace_msg
                    
                    # Request 2: Chat response via Groq
                    groq_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
                    chat_msg = "Senin için hazırladım ve çalışma alanına ekledim! İyi çalışmalar 📚"
                    if groq_key:
                        try:
                            from groq import Groq
                            client = Groq(api_key=groq_key)
                            chat_prompt = f"Kullanıcı şunu sordu: '{prompt}'. Kullanıcıya kısa, samimi, 2-3 cümlelik Türkçe bir yanıt ver. İçeriği çalışma alanına eklediğini belirt."
                            completion = client.chat.completions.create(
                                model="llama-3.3-70b-versatile",
                                messages=[{"role": "user", "content": chat_prompt}]
                            )
                            chat_msg = completion.choices[0].message.content.strip()
                        except Exception:
                            pass

                    st.session_state.messages.append({"role": "assistant", "content": chat_msg})

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
        st.markdown("### 📅 İlerleme Durumu")
        if st.session_state.goals_list:
            now_date = now_istanbul().date()
            for g in st.session_state.goals_list:
                try:
                    s_date = datetime.strptime(g['start'], "%Y-%m-%d").date()
                    e_date = datetime.strptime(g['end'], "%Y-%m-%d").date()
                    total_days = (e_date - s_date).days
                    if total_days <= 0: total_days = 1
                    passed_days = (now_date - s_date).days
                    
                    progress = int((passed_days / total_days) * 100)
                    if progress > 100: progress = 100
                    if progress < 0: progress = 0
                    
                    if progress >= 100:
                        color = "#10B981" # Green (Done)
                    elif progress >= 80:
                        color = "#F59E0B" # Orange (Near end)
                    elif passed_days > total_days:
                        color = "#EF4444" # Red (Overdue)
                    else:
                        color = "#3B82F6" # Blue (On track)

                    st.markdown(f'''
                    <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(124, 58, 237, 0.3); border-radius: 12px; padding: 15px; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <h4 style="margin: 0; color: #F8FAFC; font-size: 1.1rem;">{g['title']}</h4>
                            <span style="font-size: 0.8rem; color: #94A3B8;">{g['start']} &rarr; {g['end']}</span>
                        </div>
                        <p style="margin: 0 0 12px 0; font-size: 0.9rem; color: #CBD5E1;">{g['desc']}</p>
                        <div style="background: #0F172A; border-radius: 8px; height: 8px; width: 100%; overflow: hidden;">
                            <div style="background: {color}; width: {progress}%; height: 100%; border-radius: 8px; transition: width 0.3s ease;"></div>
                        </div>
                        <div style="text-align: right; font-size: 0.8rem; color: #94A3B8; margin-top: 4px;">% {progress} Süre Doldu</div>
                    </div>
                    ''', unsafe_allow_html=True)
                except Exception:
                    pass
        else:
            st.info("Henüz bir hedef eklemediniz.")


elif st.session_state.current_panel == "calendar":
    st.markdown("## 📅 Çalışma Takvimi")
    now = now_istanbul()
    
    # Reminder logic — 1-minute precision, toast + warning
    for session in st.session_state.schedule_list:
        try:
            s_date = datetime.strptime(session["day"], "%Y-%m-%d").date()
            if s_date == now.date():
                s_time = datetime.strptime(session["start_time"], "%H:%M:%S").time()
                diff = (datetime.combine(s_date, s_time) - now.replace(tzinfo=None)).total_seconds()
                if abs(diff) <= 60:  # Within 1 minute of start time
                    st.toast(f"⏰ Çalışma zamanı! Görev: {session['task']}", icon="⏰")
                    st.warning(f"⏰ Çalışma zamanı! **Görev: {session['task']}** — {session['start_time']}")
        except Exception:
            pass

    col_f, col_v = st.columns([1, 2])
    
    with col_f:
        with st.form("cal_form_modern"):
            c_task = st.text_input("Görev/Ders")
            c_day = st.date_input("Gün", now)
            c_start_t = st.time_input("Başlangıç Saati", value=st.session_state.cal_start_time, step=60)
            c_end_t = st.time_input("Bitiş Saati", value=st.session_state.cal_end_time, step=60)
            if st.form_submit_button("Takvime Ekle"):
                # Persist chosen times to session_state so they don't reset
                st.session_state.cal_start_time = c_start_t
                st.session_state.cal_end_time = c_end_t
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

        DAYS_TR = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
        DAY_MAP = {
            "0": "Pazartesi", "1": "Salı", "2": "Çarşamba", "3": "Perşembe",
            "4": "Cuma", "5": "Cumartesi", "6": "Pazar"
        }
        import calendar as cal_lib

        # Build grid: day -> list of task strings
        grid = {d: [] for d in DAYS_TR}
        for item in st.session_state.schedule_list:
            try:
                day_obj = datetime.strptime(item["day"], "%Y-%m-%d")
                weekday_name = DAYS_TR[day_obj.weekday()]
                start_fmt = item["start_time"][:5]  # HH:MM
                end_fmt = item["end_time"][:5]
                grid[weekday_name].append(f"{item['task']} ({start_fmt}-{end_fmt})")
            except Exception:
                pass

        # Render HTML table
        max_rows = max((len(v) for v in grid.values()), default=0)
        if max_rows == 0:
            st.info("Henüz bir çalışma oturumu eklemediniz.")
        else:
            header = "".join(f"<th style='padding:8px 12px; background:#1E293B; color:#A78BFA; border:1px solid #334155; min-width:100px;'>{d}</th>" for d in DAYS_TR)
            rows = ""
            for i in range(max_rows):
                cells = ""
                for d in DAYS_TR:
                    cell_val = grid[d][i] if i < len(grid[d]) else ""
                    bg = "#0F2238" if cell_val else "#0F172A"
                    cells += f"<td style='padding:8px 12px; background:{bg}; color:#E2E8F0; border:1px solid #1E293B; font-size:0.82rem;'>{cell_val}</td>"
                rows += f"<tr>{cells}</tr>"

            table_html = f"""
            <div style='overflow-x:auto; margin-top:8px;'>
            <table style='border-collapse:collapse; width:100%; table-layout:fixed;'>
              <thead><tr>{header}</tr></thead>
              <tbody>{rows}</tbody>
            </table>
            </div>
            """
            st.markdown(table_html, unsafe_allow_html=True)

