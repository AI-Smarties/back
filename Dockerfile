FROM python:3.12-alpine

EXPOSE 8000

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod -R 777 /app

CMD ["/app/entrypoint.sh"]
