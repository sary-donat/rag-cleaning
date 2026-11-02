"""בדיקת המערכת עם כמה שאלות. הרצה: python test_questions.py"""
from rag import ask, create_query_engine

QUESTIONS = [
    "באיזה חומר מומלץ לנקות ספת עור?",
    "איך מורידים אבנית מראש המקלחת?",
    "איך מוציאים כתם דם מבגד?",
    "האם מותר לנקות שיש עם חומץ?",
    "אילו חומרי ניקוי אסור לערבב עם אקונומיקה?",
    "מה מזג האוויר מחר בירושלים?",  # שאלה שהתשובה עליה לא נמצאת במאגר
]

engine = create_query_engine()

for i, q in enumerate(QUESTIONS, 1):
    answer, sources = ask(engine, q)
    print(f"\n{i}. שאלה: {q}")
    print(f"   תשובה: {answer}")
    print(f"   מקורות: {[s['file'] for s in sources] or 'אין'}")
