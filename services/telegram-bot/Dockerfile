FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
	apt-get upgrade -y && \
	apt-get install -y --no-install-recommends ca-certificates && \
	rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY main.py .

ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]

