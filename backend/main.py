import json
import os
import re
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from . import models, schemas, database
from .agents.graph import coaching_agent_app
from langchain_core.messages import HumanMessage

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, RedirectResponse
import io
import os
from fpdf import FPDF

# Create tables (for local sqlite dev)
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="AI Coaching Assistant API")

# Mount static files to serve index.html and other assets
# The index.html is in the parent directory relative to backend, or I can just serve it from where it is.
# Actually, I'll serve the root directory.


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(os.path.dirname(BASE_DIR), "index.html")
DEMO_PATH = os.path.join(os.path.dirname(BASE_DIR), "demo.html")

@app.get("/dashboard")
async def get_index():
    return FileResponse(INDEX_PATH)

@app.get("/demo.html")
async def get_demo():
    return FileResponse(DEMO_PATH)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
async def read_root():
    return RedirectResponse(url="/dashboard")

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
    sessions = db.query(models.StudySession).filter(models.StudySession.user_id == user_id).order_by(models.StudySession.date.desc()).limit(5).all()
    
    return schemas.DashboardData(user=user, goals=goals, recent_sessions=sessions)


from fastapi import Request
from fastapi.responses import StreamingResponse
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import cm
import io


def create_pdf_buffer(content, title="IREM AI"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=16)
    pdf.cell(0, 10, title, ln=True)
    pdf.ln(5)
    pdf.set_font("Helvetica", size=11)
    
    # Fix Turkish chars by replacing with ASCII equivalents (for Streamlit Cloud compatibility)
    replacements = {
        'ş': 's', 'Ş': 'S', 'ğ': 'g', 'Ğ': 'G',
        'ü': 'u', 'Ü': 'U', 'ö': 'o', 'Ö': 'O',
        'ı': 'i', 'İ': 'I', 'ç': 'c', 'Ç': 'C'
    }
    safe_content = content
    for tr, en in replacements.items():
        safe_content = safe_content.replace(tr, en)
    
    for line in safe_content.split('\n'):
        pdf.multi_cell(0, 7, line[:200])
    
    buffer = io.BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

@app.post("/api/generate-pdf")
async def generate_pdf_endpoint(request: Request):
    body = await request.json()
    content = body.get("content", "İçerik yok")
    title_text = body.get("title", "IREM AI - Not")
    buffer = create_pdf_buffer(content, title_text)
    return StreamingResponse(buffer, media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename=not.pdf"})


# Connection manager for websockets
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
    
    # We will use user_id as the thread_id for LangGraph memory
    thread_id = str(user_id)
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        while True:
            data = await websocket.receive_text()
            # Expecting a JSON with the message text
            try:
                msg_data = json.loads(data)
                user_message = msg_data.get("message", "")
            except:
                user_message = data
                
            # Log user message in DB
            db_chat = models.ChatHistory(user_id=user_id, session_id=thread_id, role="user", content=user_message)
            db.add(db_chat)
            db.commit()
            
            # Send message to LangGraph
            print(f"DEBUG: Processing message from user {user_id}: {user_message}")
            user_name = msg_data.get("user_name", "") if isinstance(msg_data, dict) else ""
            state_update = {
                "messages": [HumanMessage(content=user_message)],
                "user_context": f"User ID: {user_id}, name: {user_name}",
                "user_name": user_name,
                "language": "",  # Will be detected by orchestrator
            }
            
            # Stream the response
            try:
                # Use ainvoke for better async performance and to prevent hanging
                response_state = await coaching_agent_app.ainvoke(state_update, config)
                last_message = response_state["messages"][-1].content
                active_agent = response_state.get("next_agent", "SYSTEM")
                sentiment = response_state.get("sentiment", "Neutral")
                energy = response_state.get("energy_score", 5)
                print(f"DEBUG: Agent {active_agent} responded with sentiment {sentiment}")
            except Exception as e:
                print(f"Error invoking agent: {e}")
                import traceback
                traceback.print_exc()
                last_message = f"Sistem hatası oluştu. Lütfen bağlantınızı veya API anahtarlarınızı kontrol edin.\nHata detayı: {str(e)[:100]}..."
                active_agent = "SYSTEM"
                sentiment = "Error"
                energy = 0
            
            # Log agent response in DB
            try:
                db_chat_agent = models.ChatHistory(user_id=user_id, session_id=thread_id, role="assistant", content=last_message, agent_used=active_agent)
                db.add(db_chat_agent)
                db.commit()
            except Exception as db_err:
                print(f"Database Log Error: {db_err}")
            
            # Send back to client
            try:
                await manager.send_personal_message(json.dumps({
                    "agent": active_agent,
                    "message": last_message,
                    "sentiment": sentiment,
                    "energy": energy
                }), user_id)
            except Exception as ws_err:
                print(f"WebSocket Send Error: {ws_err}")
                break # Break loop if we can't send
            
    except WebSocketDisconnect:
        manager.disconnect(user_id)
