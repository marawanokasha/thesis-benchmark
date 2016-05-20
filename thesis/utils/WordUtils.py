import nltk
import string
from thesis.utils.config import app_config

STOP_WORDS = nltk.corpus.stopwords.words('english')
STOP_WORDS.extend(["'s", "n't","'t","nt","didn"])
STOP_WORDS = set(STOP_WORDS)
PUNCTUATION = list(string.punctuation)
WHITESPACE  = list(string.whitespace)

def stemtokenizer(text):
	""" MAIN FUNCTION to get clean stems out of a text. A list of clean stems are returned """
	stems = [] #result
	text = remove_digits(text)
	text = replace_any(text, PUNCTUATION, ' ')
	text = replace_any(text, WHITESPACE , ' ')
	tokens = text.split(' ')
	for token in tokens:
		if get_word_bucket(token) == token: #make sure it's not stopword, number or empty
			stem = app_config.stem(app_config.lemmatize(token.strip()))
			if stem != '':
				stems.append(stem)
	return stems

def replace_any(s,substrings,replacement):
	""" Given a string, replace all occurrences of any of the given substrings with the given replacement. New string is returned"""
	if type(substrings) == str:
		substrings = [substrings]
	
	for sub in substrings:
		s = s.replace(sub,replacement)
		
	return s


def is_number(str):
	""" Returns true if given string is a number (float or int)"""
	try:
		float(str)
		return True
	except ValueError:
		return False

def numeric_prefix(str):
	""" Returns true if given string starts with a digit """
	if str =='':
		return False
	
	return is_number(str[:1])


def remove_digits(word):
	""" Returns word with all digits removed from all positions of the string word"""
	word = replace_any(word, list(string.digits), '')
	return word

def remove_punctuation(sentence):
	""" removes punctuation of the given word """
	sentence = replace_any(sentence, PUNCTUATION, '')
	return sentence
      
def is_stopword(word):
  return word in STOP_WORDS

def filter_stopwords(tokenized_text):
  return [word for word in tokenized_text if not is_stopword(word)]

def get_word_bucket(word):
	""" Returns the name of the bucket that the given word belongs to. Like _stop_word_ or _number_...etc or the word itself """
	if word == '':
		bucket = '_empty_'
	elif is_number(word):
		bucket = '_number_'
	elif word in STOP_WORDS:
		bucket = '_stop_word_'
	else:
		bucket = word
		
	return bucket

def valid_stem(word):
	""" Returns true if the given word is a valid non-trivial english stem (in the simple sense)"""
	return len(word) > 1 and not is_number(word) and not numeric_prefix(word)
		
def get_word_set(articles, stemfunc = app_config.stem):
	""" Get a set of stemmed words in a list of articles. Punctuation characters are removed too """
	word_set = set()
	for article in articles:
		word_list = article.get_raw_words()
		for word in word_list:
			word = stemfunc(word)
			if valid_stem(word):
				word_set.add(word)
	
	return word_set

def search_word(articles,word):
	""" prints all article ids that has at least one occurence of the given word """
	result = []
	for article in articles:
		if article.has(word):
			result.append(article)
			print article.get_id()
	return result
			
def search_article(articles,srcfile,index):
	""" Given a list of articles, it looks up an article given a sourcefile name and index of article in that file"""
	for article in articles:
		if article.src == srcfile and article.index == index:
			return article
			
