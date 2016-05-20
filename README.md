## Setup

First for Scipy:

    sudo apt-get install build-essential gfortran libatlas-base-dev
    
Then:

    pip install virtualenv virtualenvwrapper
    mkvirtualenv thesis-env
    workon thesis-env
    pip install -r requirements.txt
    
For NLTK:

Go into a python console, then:
    
    workon thesis-env
    nltk.download("wordnet")
    nltk.download("stopwords")