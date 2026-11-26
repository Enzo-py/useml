FROM python:3.11-slim
RUN pip install --no-cache-dir fastapi==0.115.6 uvicorn==0.32.1
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app
COPY --chown=user app.py index.html ./
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers"]
