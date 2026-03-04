FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt pytest

COPY . .
COPY dev/week-1/homework/reference_solution/ /app/week-1/homework/reference_solution/
COPY dev/week-1/homework/hidden_tests /app/week-1/homework/hidden_tests

WORKDIR /app/week-1/homework
CMD ["pytest", "tests", "hidden_tests", "-v"]
