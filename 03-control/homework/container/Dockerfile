FROM python:3.10-slim

WORKDIR /app

COPY 03-control/homework/container/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . .
COPY 03-control/homework/container/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /app/03-control/homework
ENTRYPOINT ["/entrypoint.sh"]
CMD []
