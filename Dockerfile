FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y ffmpeg git && \
    rm -rf /var/lib/apt/lists/* \

WORKDIR /app

COPY requirement.txt ./

RUN pip install --upgrade pip && \
    pip install -r requirement.txt

COPY . .

EXPOSE 8501

ENV PYTHONUNBUFFERED=1

CMD ["streamlit", "run", "video_summary/src/video_summary/main.py"] 