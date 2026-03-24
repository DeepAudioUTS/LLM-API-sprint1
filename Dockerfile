FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

# UvicornでFastAPIを起動するコマンド
CMD ["python", "-m", "uvicorn", "story_api.main:app", "--host", "0.0.0.0", "--port", "8090", "--reload"]