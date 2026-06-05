"""
Release group (album) KNN offline training — not implemented yet.

Planned layout (mirror ml/artist/):
  config.py, data.py, features.py, train.py, artifact.py, gcs_upload.py
  scripts/train_local.py, scripts/upload_release_group.py

Planned .env vars:
  RELEASE_GROUP_MODEL_LOCAL_FILENAME
  RELEASE_GROUP_MODEL_BLOB_NAME
  RELEASE_GROUP_ML_MAX_* (optional training caps)
"""
