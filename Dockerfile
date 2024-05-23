FROM python:3.9.7 as builder

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --no-deps  --no-cache-dir -r requirements.txt #Forbbiden HASH Check temporarily and exclude cache
RUN pip uninstall -y opencv-python && \
    pip uninstall -y opencv-python-headless && \
    pip install --no-cache-dir opencv-python-headless # Replace one of dependences
RUN pip install --upgrade --no-cache-dir torch # Add necessary dependences of linux

COPY ultralytics ./ultralytics
COPY Alibaba-PuHuiTi-Bold.ttf ./
COPY Arial.ttf ./
COPY CITATION.cff ./
COPY inference.py ./
COPY MANIFEST.in ./
COPY mask.pt ./
COPY setup.cfg ./
COPY setup.py ./


FROM python:3.9.7-slim

WORKDIR /app

# Copy from last stage with necessary items
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /app /app

CMD python inference.py
