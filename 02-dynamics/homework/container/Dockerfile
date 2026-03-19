# python-fcl ships manylinux x86_64 wheels only (no aarch64); amd64 avoids source builds on Apple Silicon.
FROM --platform=linux/amd64 python:3.10-slim-bookworm

WORKDIR /app

COPY 02-dynamics/homework/container/requirements-tests.txt /tmp/requirements-tests.txt
RUN pip install --no-cache-dir -r /tmp/requirements-tests.txt

COPY . .
COPY 02-dynamics/homework/container/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /app/02-dynamics/homework
ENTRYPOINT ["/entrypoint.sh"]
CMD []
