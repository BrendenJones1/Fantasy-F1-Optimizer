FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY alembic.ini .
COPY migrations migrations
COPY f1opt f1opt
COPY reports reports
EXPOSE 8000
CMD ["uvicorn", "--factory", "f1opt.api:app_factory", "--host", "0.0.0.0", "--port", "8000"]
