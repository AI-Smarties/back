FROM python:3.12-alpine

WORKDIR /app

EXPOSE 8000

COPY . .

RUN chmod +x /app/entrypoint.sh

RUN pip install --no-cache-dir -r requirements.txt

CMD ["/app/entrypoint.sh"]
