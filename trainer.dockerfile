# Base image
FROM python:3.7-slim

# install python
RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY reports/ reports/

#install packages
WORKDIR /d
RUN pip install -r requirements.txt --no-cache-dir

#name our training script as the entrypoint for our docker image
ENTRYPOINT ["python", "-u", "src/models/train_model.py", "data/processed", "models", "reports/figures"]