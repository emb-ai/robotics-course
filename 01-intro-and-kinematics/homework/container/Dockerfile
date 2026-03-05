FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt pytest

COPY . .

WORKDIR /app/01-intro-and-kinematics/homework
CMD ["pytest", "tests", "-v"]
