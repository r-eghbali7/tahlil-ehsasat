import pandas as pd
from transformers import pipeline

print("در حال بارگذاری مدل هوش مصنوعی... (این مرحله در بار اول به دلیل دانلود مدل زمان‌بر است)")
# فراخوانی مدل دیجی‌کالا از هاگینگ‌فیس
sentiment_pipeline = pipeline("text-classification", model="HooshvareLab/bert-fa-base-uncased-sentiment-digikala")

# ۱. خواندن فایل کامنت‌ها
df = pd.read_excel('comments.xlsx')

# ۲. تعریف تابع تحلیل‌گر
def analyze_comment(text):
    # بررسی خالی نبودن متن
    if not isinstance(text, str) or len(text.strip()) == 0:
        return "نامشخص", 0.0
    
    try:
        # ارسال متن به مدل و دریافت نتیجه
        result = sentiment_pipeline(text)[0]
        raw_label = result['label']
        score = round(result['score'] * 100, 2) # درصد اطمینان مدل
        
        # مدل‌های مختلف ممکن است خروجی‌های متفاوتی مثل LABEL_0 یا کلمات انگلیسی داشته باشند
        # در اینجا خروجی را به فارسی قابل فهم ترجمه می‌کنیم
        if raw_label in ['happy', 'recommended', 'LABEL_0', 'positive']:
            sentiment = 'مثبت'
        elif raw_label in ['sad', 'not_recommended', 'angry', 'LABEL_1', 'negative']:
            sentiment = 'منفی'
        else:
            sentiment = 'خنثی'
            
        return sentiment, score
    except Exception as e:
        return "خطا", 0.0

print("در حال تحلیل کامنت‌ها...")

# ۳. اعمال مدل روی تمام ردیف‌های ستون کامنت‌ها
# zip کمک می‌کند دو ستون مجزا (احساس و درصد اطمینان) همزمان ساخته شوند
df['sentiment'], df['confidence_score'] = zip(*df['comment_text'].apply(analyze_comment))

# ۴. ذخیره نتایج در فایل جدید
df.to_excel('analyzed_comments.xlsx', index=False)

print("پردازش تمام شد! فایل analyzed_comments.xlsx را بررسی کنید.")