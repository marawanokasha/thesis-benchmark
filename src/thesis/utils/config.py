import ConfigParser
from collections import namedtuple

import nltk


def read_config():
    config_parser = ConfigParser.SafeConfigParser()
    config_parser.read("etc/config.ini")
    return config_parser


def get_app_config():
    AppConfig = namedtuple('AppConfig', ['stem', 'lemmatize', 'kfold'])

    return AppConfig(
        # stem function
        # stem = nltk.stem.lancaster.LancasterStemmer().stem
        stem = nltk.stem.porter.PorterStemmer().stem,
        # stem=nltk.stem.snowball.EnglishStemmer().stem,
        lemmatize=nltk.stem.wordnet.WordNetLemmatizer().lemmatize
        # Model config
        # kfold=10
    )


config = read_config()

app_config = get_app_config()
