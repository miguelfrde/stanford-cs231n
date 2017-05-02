# Music genre classification

## Experiment with different architectures

- Typical VGG-like architecture
- GoogLeNet
- ResNet
- GANs?
- DenseNet


## Datasets

10M Song Dataset subset for development:

```
wget http://static.echonest.com/millionsongsubset_full.tar.gz
```

10M Song Dataset:

```
rsync -avzuP publicdata.opensciencedatacloud.org::ark:/31807/osdc-c1c763e4/ /path/to/local_copy
```

## Getting started

```
# Get python 3
git clone https://github.com/miguelfrde/stanford-cs231n
pip install virtualenvwrapper
export WORKON_HOME=~/.virtualenvs
source /usr/local/bin/virtualenvwrapper.sh
cd /path/to/this/repo/project
mkvirtualenv cs231n-project -p python3
workon cs231n-project
pip install -r requirements.txt
jupyter notebook
```
