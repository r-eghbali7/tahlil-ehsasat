from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from transformers import pipeline

# ==========================================
# 1. Database Setup (SQLite + SQLAlchemy)
# ==========================================
SQLALCHEMY_DATABASE_URL = "sqlite:///./comments.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# مدل جدول دیتابیس
class DBComment(Base):
    __tablename__ = "comments"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    sentiment = Column(String, index=True)          # مثبت، منفی، خنثی
    sentiment_score = Column(Float)                 # درصد اطمینان احساس
    category = Column(String, index=True)           # انتقادی، تشکر، سوال و...
    category_score = Column(Float)                  # درصد اطمینان دسته‌بندی

Base.metadata.create_all(bind=engine)

# Dependency برای گرفتن سشن دیتابیس
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==========================================
# 2. Pydantic Schemas
# ==========================================
class CommentRequest(BaseModel):
    text: str

class CommentResponse(BaseModel):
    id: int
    text: str
    sentiment: str
    sentiment_score: float
    category: str
    category_score: float

    class Config:
        orm_mode = True

# ==========================================
# 3. AI Models Setup
# ==========================================
print("در حال بارگذاری مدل‌های هوش مصنوعی... (این کار ممکن است کمی طول بکشد)")

# مدل اول: تحلیل احساسات (دیجی‌کالا)
sentiment_pipeline = pipeline("text-classification", model="HooshvareLab/bert-fa-base-uncased-sentiment-digikala")

# مدل دوم: دسته‌بندی Zero-Shot (برای تشخیص انتقاد، سوال، شکایت)
# نکته: حجم این مدل حدود ۲ گیگابایت است و در اولین اجرا دانلود می‌شود.
zero_shot_pipeline = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")
candidate_labels = ["تشکر و تعریف", "انتقاد از محصول", "سوال درباره محصول", "شکایت از ارسال"]

# ==========================================
# 4. FastAPI Application
# ==========================================
app = FastAPI(title="سیستم هوشمند تحلیل کامنت مشتریان")

@app.post("/analyze/", response_model=CommentResponse)
def analyze_and_store_comment(comment_req: CommentRequest, db: Session = Depends(get_db)):
    text = comment_req.text
    if not text.strip():
        raise HTTPException(status_code=400, detail="متن کامنت نمی‌تواند خالی باشد")

    # --- الف) تحلیل احساسات ---
    sent_result = sentiment_pipeline(text)[0]
    raw_label = sent_result['label']
    sentiment_score = round(sent_result['score'] * 100, 2)
    
    if raw_label in ['happy', 'recommended', 'LABEL_0', 'positive']:
        sentiment_fa = 'مثبت'
    elif raw_label in ['sad', 'not_recommended', 'angry', 'LABEL_1', 'negative']:
        sentiment_fa = 'منفی'
    else:
        sentiment_fa = 'خنثی'

    # --- ب) دسته‌بندی موضوعی (Zero-Shot) ---
    zs_result = zero_shot_pipeline(text, candidate_labels)
    # بهترین دسته‌بندی پیدا شده
    category = zs_result['labels'][0] 
    category_score = round(zs_result['scores'][0] * 100, 2)

    # --- ج) ذخیره در دیتابیس SQLite ---
    new_comment = DBComment(
        text=text,
        sentiment=sentiment_fa,
        sentiment_score=sentiment_score,
        category=category,
        category_score=category_score
    )
    
    db.add(new_comment)
    db.commit()
    db.refresh(new_comment)

    return new_comment