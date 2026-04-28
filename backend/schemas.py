from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class UserBase(BaseModel):
    name: str
    email: str
    grade_level: str

class UserCreate(UserBase):
    pass

class User(UserBase):
    id: int
    created_at: datetime
    class Config:
        from_attributes = True

class GoalBase(BaseModel):
    title: str
    description: str
    target_date: datetime
    status: str
    agent: str

class GoalCreate(GoalBase):
    pass

class Goal(GoalBase):
    id: int
    user_id: int
    class Config:
        from_attributes = True

class ChatMessage(BaseModel):
    role: str
    content: str
    agent: Optional[str] = None
    
class ChatRequest(BaseModel):
    session_id: str
    user_id: int
    message: str
    
class DashboardData(BaseModel):
    user: User
    goals: List[Goal]
    recent_sessions: list
