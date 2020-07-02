#!/bin/bash

sudo apt-get -y update
sudo apt-get -y upgrade

# install accelerated BLAS environment 
# openBLAS should be there by default...just in case
sudo apt-get -y install vim git build-essential libblas-dev liblapack-dev libatlas-base-dev gfortran python-dev libfreetype6-dev libxft-dev awscli gtk2-engines-pixbuf
export LAPACK=/usr/lib/liblapack.so
export ATLAS=/usr/lib/libatlas.so

# more required dependencies
sudo apt-get install -y default-mysql-server libmariadb-dev-compat libmariadb-dev python-tk

# create virtual_env and install requirements
cd TwitterLDATopicModeling/
wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
rm get-pip.py
sudo pip install virtualenv

virtualenv -p /usr/bin/python2.7 py2_enviro
source py2_enviro/bin/activate
pip install -r requirements.txt
sudo activate-global-python-argcomplete

# stopword list using nltk brown corpora 
python patches/nltk_downloads.py
mv patches/english ~/nltk_data/corpora/stopwords

# custom progress bar patched
sudo cp patches/prog_class.py py2_enviro/lib/python2.7/site-packages/pyprind/prog_class.py
