FROM python:3.10-slim

WORKDIR /app

ARG INSTALL_CUDA_TORCH=0
COPY 01-intro-and-kinematics/homework/container/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && if [ "$INSTALL_CUDA_TORCH" = "1" ]; then \
         pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124; \
       else \
         pip install --no-cache-dir torch; \
       fi

COPY . .
COPY 01-intro-and-kinematics/homework/container/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /app/01-intro-and-kinematics/homework
ENTRYPOINT ["/entrypoint.sh"]
CMD []
