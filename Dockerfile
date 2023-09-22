FROM python:3.11.5

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt
RUN python -m nltk.downloader stopwords
RUN pip install torch

WORKDIR /app

COPY . /app

CMD ["uvicorn", "app.app:app", "--host=0.0.0.0", "--port=8084"]