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