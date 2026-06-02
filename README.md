# rec_o

## Set up your machine
### Virtualenv
```
cd ~/code
mkdir GuerillaUmNeon
git clone git@github.com:GuerillaUmNeon/rec_o.git
cd rec_o
pyenv install 3.13.13
pyenv virtualenv 3.13.13 rec-o-env
pyenv local rec-o-env
```

### Requirements
```
pip install --upgrade pip
pip install -r requirements.txt
pip freeze
```

## FastAPI
### Test
```
uvicorn app.main:app --reload
```
* [http://localhost:8000](http://localhost:8000)
* [http://localhost:8000/predict](http://localhost:8000/predict)
* [http://localhost:8000/docs](http://localhost:8000/docs)

### Prod
* https://rec-o-api-779590423195.europe-west1.run.app/
* https://rec-o-api-779590423195.europe-west1.run.app/docs#/

## Docker
```
docker build -t rec-o .
docker run --name rec-o-api -p 8000:8000 --env-file .env rec-o
```

Delete if needed:
```
docker rm rec-o-api
```

## .env
Token was generated with:
```
python -c "import secrets; print(secrets.token_hex(32))"
```

## GCP
APIs activated for project:
* Cloud Build API
* Artifact Registry API
* Cloud Run Admin API
* Secret Manager API