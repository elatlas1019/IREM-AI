from sqlalchemy.orm import Session
from . import models, schemas
from datetime import datetime

def get_goals(db: Session, user_id: int):
    return db.query(models.Goal).filter(models.Goal.user_id == user_id).all()

def create_goal(db: Session, goal: schemas.GoalCreate, user_id: int):
    db_goal = models.Goal(
        user_id=user_id,
        title=goal.title,
        description=goal.description,
        target_date=goal.target_date,
        status=goal.status,
        agent=goal.agent
    )
    db.add(db_goal)
    db.commit()
    db.refresh(db_goal)
    return db_goal

def delete_goal(db: Session, goal_id: int):
    db_goal = db.query(models.Goal).filter(models.Goal.id == goal_id).first()
    if db_goal:
        db.delete(db_goal)
        db.commit()
        return True
    return False

def get_study_sessions(db: Session, user_id: int):
    return db.query(models.StudySession).filter(models.StudySession.user_id == user_id).all()

def create_study_session(db: Session, session_data: dict, user_id: int):
    db_session = models.StudySession(
        user_id=user_id,
        duration_minutes=session_data['duration_minutes'],
        subject=session_data['subject'],
        date=session_data['date']
    )
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session
