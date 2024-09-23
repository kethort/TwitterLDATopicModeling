#!/bin/bash

sudo apt-get -y update
sudo apt-get -y upgrade

# install accelerated BLAS environment 
# openBLAS should be there by default...just in case
sudo apt-get -y install vim git build-essential libblas-dev liblapack-dev libatlas-base-dev gfortran libfreetype6-dev libxft-dev gtk2-engines-pixbuf
export LAPACK=/usr/lib/liblapack.so
export ATLAS=/usr/lib/libatlas.so

# more required dependencies
sudo apt-get install -y default-mysql-server libmariadb-dev-compat libmariadb-dev python3-tk python3.12-venv liblzma-dev libsqlite3-dev libbz2-dev 

# create virtual_env and install requirements
cd TwitterLDATopicModeling/

python3 -m venv py312_enviro
source py312_enviro/bin/activate
pip install -r requirements_py3.txt
activate-global-python-argcomplete

# stopword list using nltk brown corpora 
python patches/nltk_downloads.py
cp patches/english ~/nltk_data/corpora/stopwords

# custom progress bar patched
sudo cp patches/prog_class.py py312_enviro/lib/python3.12/site-packages/pyprind/prog_class.py
