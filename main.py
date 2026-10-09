from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # اضافه شده برای اتصال فرانت
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from transformers import pipeline
from typing import List  # اضافه شده

# ==========================================
# 1. Database Setup
# ==========================================
SQLALCHEMY_DATABASE_URL = "sqlite:///./comments.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class DBComment(Base):
    __tablename__ = "comments"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    sentiment = Column(String, index=True)
    sentiment_score = Column(Float)
    category = Column(String, index=True)
    category_score = Column(Float)

Base.metadata.create_all(bind=engine)

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
print("در حال بارگذاری مدل‌های هوش مصنوعی...")
sentiment_pipeline = pipeline("text-classification", model="HooshvareLab/bert-fa-base-uncased-sentiment-digikala")
zero_shot_pipeline = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")
candidate_labels = ["تشکر و تعریف", "انتقاد از محصول", "سوال درباره محصول", "شکایت از ارسال"]

# ==========================================
# 4. FastAPI Application
# ==========================================
app = FastAPI(title="سیستم هوشمند تحلیل کامنت مشتریان")

# تنظیمات CORS برای اجازه دادن به index.html برای ارسال درخواست
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # در محیط واقعی آدرس دامین فرانت را بگذارید
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze/", response_model=CommentResponse)
def analyze_and_store_comment(comment_req: CommentRequest, db: Session = Depends(get_db)):
    text = comment_req.text
    if not text.strip():
        raise HTTPException(status_code=400, detail="متن کامنت نمی‌تواند خالی باشد")

    sent_result = sentiment_pipeline(text)[0]
    raw_label = sent_result['label']
    sentiment_score = round(sent_result['score'] * 100, 2)

    if raw_label in ['happy', 'recommended', 'LABEL_0', 'positive']:
        sentiment_fa = 'مثبت'
    elif raw_label in ['sad', 'not_recommended', 'angry', 'LABEL_1', 'negative']:
        sentiment_fa = 'منفی'
    else:
        sentiment_fa = 'خنثی'

    zs_result = zero_shot_pipeline(text, candidate_labels)
    category = zs_result['labels'][0] 
    category_score = round(zs_result['scores'][0] * 100, 2)

    new_comment = DBComment(
        text=text, sentiment=sentiment_fa, sentiment_score=sentiment_score,
        category=category, category_score=category_score
    )

    db.add(new_comment)
    db.commit()
    db.refresh(new_comment)
    return new_comment

# مسیر جدید برای پنل گزارش‌گیری
@app.get("/reports/", response_model=List[CommentResponse])
def get_all_reports(db: Session = Depends(get_db)):
    return db.query(DBComment).order_by(DBComment.id.desc()).all()



# اضافه کردن این کلاس به بخش 2 (Pydantic Schemas)
class ExecutiveSummaryResponse(BaseModel):
    total_comments: int
    positive_percent: float
    negative_percent: float
    neutral_percent: float
    dominant_category: str
    recommendation: str
    ai_analysis: str


# اضافه کردن این API به انتهای فایل (بخش 4)
@app.get("/summary/", response_model=ExecutiveSummaryResponse)
def get_executive_summary(db: Session = Depends(get_db)):
    comments = db.query(DBComment).all()
    total = len(comments)
    
    if total == 0:
        return ExecutiveSummaryResponse(
            total_comments=0, positive_percent=0, negative_percent=0, neutral_percent=0,
            dominant_category="نامشخص", recommendation="نامشخص", ai_analysis="هنوز داده‌ای برای تحلیل وجود ندارد."
        )

    # شمارش احساسات
    pos_count = sum(1 for c in comments if c.sentiment == 'مثبت')
    neg_count = sum(1 for c in comments if c.sentiment == 'منفی')
    neu_count = sum(1 for c in comments if c.sentiment == 'خنثی')

    # یافتن موضوع پرتکرار
    category_counts = {}
    for c in comments:
        category_counts[c.category] = category_counts.get(c.category, 0) + 1
    dominant_category = max(category_counts, key=category_counts.get)

    # محاسبه درصدها
    pos_pct = round((pos_count / total) * 100, 1)
    neg_pct = round((neg_count / total) * 100, 1)
    neu_pct = round((neu_count / total) * 100, 1)

    # ==========================================
    # موتور تصمیم‌ساز (Decision Engine)
    # ==========================================
    if pos_pct >= 60:
        recommendation = "بله، محصول شارژ شود ✅"
        ai_analysis = f"وضعیت بسیار عالی است! {pos_pct}٪ از مشتریان راضی هستند. پرتکرارترین موضوع '{dominant_category}' بوده است. محصول پتانسیل فروش بالایی دارد."
    elif neg_pct >= 40:
        recommendation = "خیر، توقف موقت فروش ❌"
        ai_analysis = f"هشدار! {neg_pct}٪ از بازخوردها منفی است. مشکل اصلی در زمینه '{dominant_category}' است. پیشنهاد می‌شود قبل از شارژ مجدد، این مشکل برطرف شود."
    else:
        recommendation = "نیاز به بررسی دستی ⚠️"
        ai_analysis = f"نظرات متناقض است (مثبت: {pos_pct}٪، منفی: {neg_pct}٪). وضعیت '{dominant_category}' بیشترین بحث را داشته است. بهتر است با احتیاط شارژ شود."

    return ExecutiveSummaryResponse(
        total_comments=total,
        positive_percent=pos_pct,
        negative_percent=neg_pct,
        neutral_percent=neu_pct,
        dominant_category=dominant_category,
        recommendation=recommendation,
        ai_analysis=ai_analysis
    )