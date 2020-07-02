#!/bin/bash

sudo apt-get -y update
sudo apt-get -y upgrade

# install accelerated BLAS environment 
# openBLAS should be there by default...just in case
sudo apt-get -y install vim git build-essential libblas-dev liblapack-dev libatlas-base-dev gfortran python-dev libfreetype6-dev libxft-dev awscli gtk2-engines-pixbuf
export LAPACK=/usr/lib/liblapack.so
export ATLAS=/usr/lib/libatlas.so

# more required dependencies
sudo apt-get install -y default-mysql-server libmariadb-dev-compat libmariadb-dev python-tk liblzma-dev libsqlite3-dev libbz2-dev 

# build python3.6 from source
wget https://www.python.org/ftp/python/3.6.0/Python-3.6.0.tar.xz
tar xJf Python-3.6.0.tar.xz
cd Python-3.6.0

sudo ./configure
sudo make
sudo make install

# create virtual_env and install requirements
cd TwitterLDATopicModeling/
wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
rm get-pip.py
sudo pip install virtualenv

virtualenv -p /usr/local/bin/python3.6 py3_enviro
source py3_enviro/bin/activate
pip install -r requirements_py3.txt
sudo activate-global-python-argcomplete

# stopword list using nltk brown corpora 
python patches/nltk_downloads.py
cp patches/english ~/nltk_data/corpora/stopwords

# custom progress bar patched
sudo cp patches/prog_class.py py3_enviro/lib/python3.6/site-packages/pyprind/prog_class.py