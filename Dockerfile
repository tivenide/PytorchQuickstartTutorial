FROM nvidia/cuda:12.1.0-base-ubuntu22.04

RUN apt-get update &&  \
    apt-get install --no-install-recommends -y python3-pip python3-dev ffmpeg libsm6 libxext6 gcc g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /PytorchQuickstartTutorial

COPY requirements.txt ./

RUN pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121 --verbose

COPY . ./

CMD ["python3", "main.py"]