import os
from celery import Celery
from celery.schedules import crontab

redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "hakitrade_worker",
    broker=redis_url,
    backend=redis_url,
    include=["app.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Kolkata",
    enable_utc=True,
)

# Schedule the EOD data ingestion at 4:15 PM IST daily
celery_app.conf.beat_schedule = {
    "fetch-eod-data-daily": {
        "task": "app.tasks.fetch_daily_market_data",
        "schedule": crontab(hour=16, minute=15), # 16:15 IST
    },
}
