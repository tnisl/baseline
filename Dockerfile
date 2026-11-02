FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY umilife /app/umilife
COPY run.py /app/run.py
COPY models /app/models
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
ENTRYPOINT ["python","/app/run.py"]
CMD ["--data-root","/data","--split","private_test","--model","/app/models/qwen","--adapter-dir","/app/models/adapters","--asr-model","/app/models/whisper","--output","/app/result/submission.jsonl","--fp16"]
