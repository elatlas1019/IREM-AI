import streamlit as st
import os
import json
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from backend.agents.graph import coaching_agent_app
from backend.main import create_pdf_buffer

# Load env
load_dotenv()

# --- Page Config ---
st.set_page_config(
    page_title="IREM AI - Smart Educational Coach",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEME COLORS & CSS (Directly from index.html) ---
st.markdown("""
    <style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        background: #0F172A !important;
        font-family: 'Inter', sans-serif !important;
        color: #F1F5F9 !important;
    }
    
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    [data-testid="stSidebar"] {
        background: #111827 !important;
        border-right: 1px solid rgba(124, 58, 237, 0.2);
        width: 200px !important;
    }
    
    /* Sidebar Components */
    .logo-text {
        font-size: 1.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #7C3AED, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        display: block;
    }
    
    .nav-btn {
        display: block;
        width: 100%;
        padding: 12px 16px;
        margin: 4px 0;
        background: transparent;
        border: none;
        color: #94A3B8;
        text-align: left;
        border-radius: 12px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s;
    }
    .nav-btn.active {
        background: rgba(124, 58, 237, 0.15);
        color: #C084FC;
    }
    
    /* Daily Quote Card (The Neon One) */
    .quote-banner {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.2), rgba(236, 72, 153, 0.15));
        border: 1px solid rgba(236, 72, 153, 0.5);
        box-shadow: 0 0 20px rgba(124, 58, 237, 0.3), inset 0 0 15px rgba(236, 72, 153, 0.2);
        border-radius: 20px;
        padding: 20px 30px;
        text-align: center;
        margin-bottom: 2rem;
        animation: float 4s ease-in-out infinite;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    /* Chat & Workspace Layout */
    .stChatFloatingInputContainer { background: transparent !important; }
    
    .chat-bubble {
        padding: 1rem 1.2rem;
        border-radius: 18px;
        margin-bottom: 1rem;
        max-width: 85%;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .user-msg {
        background: linear-gradient(135deg, #7C3AED, #4F46E5);
        margin-left: auto;
        border-bottom-right-radius: 4px;
        color: white;
    }
    .assistant-msg {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(124, 58, 237, 0.2);
        margin-right: auto;
        border-bottom-left-radius: 4px;
    }
    
    .doc-panel {
        background: #1E293B;
        border-radius: 24px;
        padding: 2rem;
        border: 1px solid rgba(124, 58, 237, 0.1);
        height: 600px;
        overflow-y: auto;
    }
    
    /* Energy Bar */
    .energy-bar-bg {
        background: #334155;
        height: 6px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .energy-bar-fill {
        background: linear-gradient(90deg, #7C3AED, #F472B6);
        height: 100%;
        border-radius: 10px;
        width: 90%; /* Placeholder for 9/10 */
    }
    </style>
""", unsafe_allow_html=True)

# --- State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_name" not in st.session_state:
    st.session_state.user_name = "brc"
if "current_panel" not in st.session_state:
    st.session_state.current_panel = "dashboard"

# --- Sidebar Rebuild ---
with st.sidebar:
    st.markdown('<span class="logo-text">💠 IREM AI</span>', unsafe_allow_html=True)
    
    # Navigation Simulation
    if st.button("🏠 Dashboard", key="btn_dash", use_container_width=True):
        st.session_state.current_panel = "dashboard"
    if st.button("🎯 Hedeflerim", key="btn_goals", use_container_width=True):
        st.session_state.current_panel = "goals"
    if st.button("📅 Takvim", key="btn_cal", use_container_width=True):
        st.session_state.current_panel = "calendar"
    if st.button("📊 Analitik", key="btn_ana", use_container_width=True):
        st.session_state.current_panel = "analytics"
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Energy Widget
    st.markdown("""
        <div style="font-size:0.82rem; color:#94A3B8;">
            <div style="display:flex; justify-content:space-between;">
                <span>Enerji</span>
                <span>9/10</span>
            </div>
            <div class="energy-bar-bg"><div class="energy-bar-fill"></div></div>
            <p style="color:#10B981; font-size:0.72rem;">✨ Ruh Hali: Pozitif</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # User Profile
    st.markdown(f"""
        <div style="display:flex; align-items:center; gap:12px; padding:10px;">
            <div style="width:36px; height:36px; background:#7C3AED; border-radius:50%; display:flex; align-items:center; justify-content:center; font-weight:800;">B</div>
            <div>
                <p style="font-size:0.88rem; font-weight:600; margin:0;">{st.session_state.user_name}</p>
                <p style="font-size:0.72rem; color:#10B981; margin:0;">● Online</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

# --- Main Logic ---
if st.session_state.current_panel == "dashboard":
    # Header
    col_h1, col_h2 = st.columns([1, 1])
    with col_h1:
        st.markdown('<h1 style="margin-bottom:0;">Hoş Geldin! 👋</h1>', unsafe_allow_html=True)
        st.markdown('<p style="color:#94A3B8; margin-top:0;">Koçunla sohbet et, planını oluştur, hedeflerini takip et!</p>', unsafe_allow_html=True)
    
    # Daily Quote Banner
    st.markdown("""
        <div class="quote-banner">
            <div style="font-size:11px; color:#F472B6; font-weight:800; letter-spacing:2px; margin-bottom:10px; text-transform:uppercase;">✨ GÜNÜN SÖZÜ</div>
            <div style="font-size:1.2rem; font-weight:700; color:white; line-height:1.5;">"Sabır en sessiz ama en güçlü öğretmendir."</div>
            <div style="font-size:0.95rem; color:#F472B6; margin-top:10px;">— İrem AI 💜</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Workspace
    col_chat, col_doc = st.columns([1.1, 1.4], gap="large")
    
    with col_chat:
        # Chat Messages
        chat_sub = st.container(height=500, border=False)
        with chat_sub:
            for m in st.session_state.messages:
                cls = "user-msg" if m["role"] == "user" else "assistant-msg"
                st.markdown(f'<div class="chat-bubble {cls}">{m["content"]}</div>', unsafe_allow_html=True)

        # Chat Input Area (Simulated)
        if prompt := st.chat_input("Zihin koçun burada, hadi konuşalım..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Agent logic
            config = {"configurable": {"thread_id": "st-session-1"}}
            state_update = {"messages": [HumanMessage(content=prompt)], "user_name": st.session_state.user_name, "language": "tr"}
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response_state = loop.run_until_complete(coaching_agent_app.ainvoke(state_update, config))
            
            last_msg = response_state["messages"][-1].content
            active_agent = response_state.get("next_agent", "SYSTEM")
            
            st.session_state.messages.append({"role": "assistant", "content": last_msg, "agent": active_agent})
            st.rerun()

    with col_doc:
        st.markdown('<div class="doc-panel">', unsafe_allow_html=True)
        st.markdown('### Çalışma Alanı')
        
        last_teach = next((m for m in reversed(st.session_state.messages) if m.get("agent") in ["TEACH", "PLAN"]), None)
        if last_teach:
            st.markdown(last_teach["content"])
            # PDF Button
            pdf_data = create_pdf_buffer(last_teach["content"], f"IREM AI - {last_teach['agent']}")
            st.download_button("📄 PDF AI", data=pdf_data, file_name="irem_not.pdf", mime="application/pdf")
        else:
            st.markdown("*Ders notları ve testler burada görünecek.*")
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.title(f"{st.session_state.current_panel.capitalize()}")
    st.info("Bu bölüm yapım aşamasındadır.")
