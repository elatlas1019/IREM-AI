import json
import os
import re
import io
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, RedirectResponse
from sqlalchemy.orm import Session

from . import models, schemas, database
from .agents.graph import coaching_agent_app
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# Initialize database tables
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="AI Coaching Assistant API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- TEXT REPORT GENERATION (Clean & Robust) ---
def create_pdf_buffer(content: str, title: str = "IREM AI") -> bytes:
    import textwrap
    
    def clean(text):
        if not text: return ""
        # Force ASCII for download compatibility, though utf-8 is fine for txt
        return text.encode('ascii', 'ignore').decode('ascii')
    
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"  {clean(title)}")
    lines.append(f"{'='*60}")
    lines.append("")
    
    for para in clean(content).split('\n'):
        if para.strip():
            wrapped = textwrap.wrap(para, width=80)
            lines.extend(wrapped)
        else:
            lines.append("")
    
    full_text = '\n'.join(lines)
    return full_text.encode('utf-8')

# --- ROUTES ---

@app.get("/")
async def read_root():
    return {"message": "AI Coaching Assistant API is running"}

@app.post("/api/users", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(database.get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        return db_user
    new_user = models.User(name=user.name, email=user.email, grade_level=user.grade_level)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.get("/api/dashboard/{user_id}", response_model=schemas.DashboardData)
def get_dashboard(user_id: int, db: Session = Depends(database.get_db)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    goals = db.query(models.Goal).filter(models.Goal.user_id == user_id).all()
    recent_sessions = db.query(models.StudySession).filter(models.StudySession.user_id == user_id).order_by(models.StudySession.date.desc()).limit(5).all()
    return schemas.DashboardData(user=user, goals=goals, recent_sessions=recent_sessions)

@app.post("/api/generate-pdf")
async def generate_pdf_endpoint(request: Request):
    body = await request.json()
    content = body.get("content", "İçerik yok")
    title_text = body.get("title", "IREM AI - Not")
    buffer_bytes = create_pdf_buffer(content, title_text)
    return StreamingResponse(io.BytesIO(buffer_bytes), media_type="text/plain", headers={"Content-Disposition": f"attachment; filename=not.txt"})

# --- WEBSOCKET CHAT ---

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[int, WebSocket] = {}

    async def connect(self, websocket: WebSocket, user_id: int):
        await websocket.accept()
        self.active_connections[user_id] = websocket

    def disconnect(self, user_id: int):
        if user_id in self.active_connections:
            del self.active_connections[user_id]

    async def send_personal_message(self, message: str, user_id: int):
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/chat/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: int, db: Session = Depends(database.get_db)):
    await manager.connect(websocket, user_id)
    thread_id = str(user_id)
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg_data = json.loads(data)
                user_message = msg_data.get("message", "")
            except:
                user_message = data
                
            # Log user message
            db_chat = models.ChatHistory(user_id=user_id, session_id=thread_id, role="user", content=user_message)
            db.add(db_chat)
            db.commit()
            
            # Agent Response
            user_name = msg_data.get("user_name", "") if isinstance(msg_data, dict) else ""
            state_update = {
                "messages": [HumanMessage(content=user_message)],
                "user_context": f"User ID: {user_id}, name: {user_name}",
                "user_name": user_name,
            }
            
            try:
                response_state = await coaching_agent_app.ainvoke(state_update, config)
                last_message = response_state["messages"][-1].content
                active_agent = response_state.get("next_agent", "SYSTEM")
                sentiment = response_state.get("sentiment", "Neutral")
                energy = response_state.get("energy_score", 5)
            except Exception as e:
                last_message = f"Sistem hatası: {str(e)[:100]}..."
                active_agent = "SYSTEM"
                sentiment = "Error"
                energy = 0
            
            # Log agent response
            db_chat_agent = models.ChatHistory(user_id=user_id, session_id=thread_id, role="assistant", content=last_message, agent_used=active_agent)
            db.add(db_chat_agent)
            db.commit()
            
            # Send to client
            await manager.send_personal_message(json.dumps({
                "agent": active_agent,
                "message": last_message,
                "sentiment": sentiment,
                "energy": energy
            }), user_id)
            
    except WebSocketDisconnect:
        manager.disconnect(user_id)
