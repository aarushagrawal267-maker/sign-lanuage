# Hugging Face Docker Space: Sign Intelligence Studio (public read-only demo).
# Built by scripts/build_hf_space.py, which stages backend, built frontend,
# MediaPipe model, trained checkpoints and the results database.
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends libegl1 libgles2 libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r backend/requirements.txt

COPY --chown=user . .

USER user
ENV SLR_PUBLIC_DEMO=true \
    SLR_DEVICE=cpu \
    SLR_DATA_DIR=/home/user/app/data \
    SLR_ARTIFACTS_DIR=/home/user/app/artifacts \
    SLR_ASSETS_DIR=/home/user/app/assets \
    SLR_FRONTEND_DIST=/home/user/app/frontend/dist \
    GLOG_minloglevel=2

WORKDIR /home/user/app/backend
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
