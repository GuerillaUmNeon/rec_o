FROM python:3.13.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --root-user-action=ignore --no-cache-dir --upgrade pip \
    && pip install --root-user-action=ignore --no-cache-dir -r requirements.txt

COPY app/ ./app/

# Minimal ml/ package so joblib can unpickle release_group artifacts
# (ListToSparseTransformer is referenced as ml.release_group.features.*).
COPY ml/__init__.py ./ml/
COPY ml/release_group/__init__.py ./ml/release_group/
COPY ml/release_group/features.py ./ml/release_group/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
