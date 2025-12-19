FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# If you’re running the bot as a worker:
CMD ["python", "gexbot_v8.py"]
