FROM python:3.10-slim

WORKDIR /app

COPY 01-intro-and-kinematics/homework/container/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . .
COPY 01-intro-and-kinematics/homework/container/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /app/01-intro-and-kinematics/homework
ENTRYPOINT ["/entrypoint.sh"]
CMD []
