FROM python:3.13.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --root-user-action=ignore --no-cache-dir --upgrade pip \
    && pip install --root-user-action=ignore --no-cache-dir -r requirements.txt

COPY app ./app/

# Minimal ml/ packages so joblib can unpickle the trained artifacts
# (pipeline steps reference ml.artist.* and ml.release_group.*).
COPY ml/__init__.py ./ml/
COPY ml/artist/__init__.py ./ml/artist/
COPY ml/artist/config.py ./ml/artist/
COPY ml/artist/features.py ./ml/artist/
COPY ml/artist/train.py ./ml/artist/
COPY ml/release_group/__init__.py ./ml/release_group/
COPY ml/release_group/config.py ./ml/release_group/
COPY ml/release_group/features.py ./ml/release_group/
COPY ml/release_group/train.py ./ml/release_group/
COPY models/knn_baseline_model_artist_full_dataset.pkl ./models/
COPY models/knn_model_release_group_full_dataset.pkl ./models/
COPY .env .

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
