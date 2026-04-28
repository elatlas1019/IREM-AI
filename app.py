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
    page_title="IREM AI - Akıllı Eğitim Koçu",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f1f5f9;
    }
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95);
        border-right: 1px solid rgba(124, 58, 237, 0.2);
    }
    .chat-bubble {
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        max-width: 85%;
        line-height: 1.6;
    }
    .user-bubble {
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        align-self: flex-end;
        margin-left: auto;
        border-bottom-right-radius: 2px;
    }
    .assistant-bubble {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(124, 58, 237, 0.2);
        border-bottom-left-radius: 2px;
    }
    .agent-tag {
        font-size: 0.7rem;
        font-weight: 800;
        text-transform: uppercase;
        color: #a78bfa;
        margin-bottom: 0.5rem;
        display: block;
    }
    h1, h2, h3 { color: #f1f5f9 !important; }
    .stButton>button {
        background: linear-gradient(135deg, #7c3aed, #ec4899);
        color: white;
        border: none;
        border-radius: 10px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(124, 58, 237, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "goals" not in st.session_state:
    st.session_state.goals = []
if "user_name" not in st.session_state:
    st.session_state.user_name = "Misafir"

# --- Sidebar ---
with st.sidebar:
    st.title("💠 IREM AI")
    st.markdown("---")
    
    # User Profile
    name = st.text_input("Adın Soyadın", value=st.session_state.user_name)
    if name != st.session_state.user_name:
        st.session_state.user_name = name
    
    st.markdown("---")
    st.subheader("🎯 Günlük Hedefler")
    new_goal = st.text_input("Yeni Hedef Ekle", key="goal_input")
    if st.button("Ekle"):
        if new_goal:
            st.session_state.goals.append({"title": new_goal, "done": False})
            st.rerun()
            
    for i, goal in enumerate(st.session_state.goals):
        checked = st.checkbox(goal["title"], value=goal["done"], key=f"goal_{i}")
        st.session_state.goals[i]["done"] = checked

# --- Main Layout ---
col1, col2 = st.columns([1.2, 0.8])

with col1:
    st.title(f"Hoş Geldin, {st.session_state.user_name}! 👋")
    st.info("İrem ile sohbet et, ders planını oluştur veya konu tekrarı yap.")

    # Chat Container
    chat_container = st.container(height=500)
    for msg in st.session_state.messages:
        with chat_container:
            role_class = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
            agent_tag = f'<span class="agent-tag">{msg["agent"]}</span>' if msg["role"] == "assistant" else ""
            st.markdown(f"""
                <div class="chat-bubble {role_class}">
                    {agent_tag}
                    {msg["content"]}
                </div>
            """, unsafe_allow_html=True)

    # Chat Input
    if prompt := st.chat_input("Zihin koçun burada, hadi konuşalım..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt, "agent": "USER"})
        
        # Call LangGraph
        config = {"configurable": {"thread_id": "st-session-1"}}
        state_update = {
            "messages": [HumanMessage(content=prompt)],
            "user_name": st.session_state.user_name,
            "language": "tr", # Default to TR
        }
        
        with st.spinner("İrem düşünüyor..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response_state = loop.run_until_complete(coaching_agent_app.ainvoke(state_update, config))
            
            last_msg = response_state["messages"][-1].content
            active_agent = response_state.get("next_agent", "SYSTEM")
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": last_msg, 
                "agent": active_agent
            })
            st.rerun()

with col2:
    st.subheader("📂 Çalışma Alanı")
    # Show the latest TEACH or PLAN content in a nice card
    last_doc = next((m for m in reversed(st.session_state.messages) if m["agent"] in ["TEACH", "PLAN"]), None)
    
    if last_doc:
        st.markdown(f"### {last_doc['agent']}")
        st.markdown(last_doc['content'])
        
        # PDF Export
        pdf_buffer = create_pdf_buffer(last_doc['content'], f"IREM AI - {last_doc['agent']}")
        st.download_button(
            label="📄 PDF Olarak İndir",
            data=pdf_buffer,
            file_name=f"irem_ai_{last_doc['agent'].lower()}.pdf",
            mime="application/pdf"
        )
    else:
        st.markdown("""
            <div style="text-align:center; padding: 2rem; background: rgba(255,255,255,0.05); border-radius: 20px;">
                <p style="font-size: 3rem;">📂</p>
                <p>Henüz bir çalışma notu oluşturulmadı.</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📅 Akademik Takvim")
    st.date_input("Çalışma Günü Seç")
    st.time_input("Başlangıç Saati")
