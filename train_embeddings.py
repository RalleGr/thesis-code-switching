#############################################################
# Source code for paper : A two-step training approach for semi-supervised language identification in code-switched data
# Authors: Dana-Maria Iliescu, Rasmus Grand & Sara Qirko
# IT-University of Copenhagen
# May 2021
#############################################################

from tools.utils import write_dict
from tools.utils import is_other
from tools.utils import print_status
from tools.utils import read_lince_labeled_data
from tools.utils import save_train_data
import os
import pandas as pd
import importlib
import sys
import fasttext
from bs4 import BeautifulSoup
import spacy
import random
import nltk
nltk.download('punkt')

WORD_DICTIONARIES_PATH = "./dictionaries/word-level/"

def create_corpus_articles(lang1, lang2, langs, embeddings_filepath, corpus_filepath):
	print_status("Creating one space-separated text file from concatenated wikipedia articles...")

	out = open(corpus_filepath, 'w', encoding='utf-8')
	sentences = []
	for lang_code in [lang1, lang2]:
		lang_name = langs[lang_code]['name']
		print_status(lang_code)
		# Load data
		for root, dirs, files in os.walk('datasets/monolingual-' + lang_code):
			if ('.DS_Store' in files):
				files.remove('.DS_Store')
			for f in files:
				print_status(f)
				filepath = os.path.join(root, f)
				file = open(filepath, 'rt', encoding='utf8')
				text = file.read()
				file.close()

				# Clean XML tags
				cleantext = BeautifulSoup(text, "lxml").text

				try:
					if (lang_code == 'arz'): lang_code = 'ar'
					module = importlib.import_module("spacy.lang." + lang_code)
					nlp = getattr(module, lang_name)()
				except:
					nlp = spacy.language.Language()
				tokenizer = nlp.Defaults.create_tokenizer(nlp)
				tokens = list(tokenizer(cleantext)) 
				s = []
				for t in tokens:
					s.append(t.text)
				s = " ".join(s)
				sentences = sentences + nltk.tokenize.sent_tokenize(s) 

	print("Sentences shuffled length: ")
	print(len(sentences))
	random.shuffle(sentences)
	for s in sentences:
		words = s.split()
		for w in words:
			w = w.lower()
			out.write(w + ' ')
		out.write("\n")

	print_status('Data saved at: ' + corpus_filepath)

def create_corpus_tweets(lang1, lang2, langs, corpus_filepath, dataset_filepath):
	print_status("Creating one space-separated text file from concatenated tweets...")

	out = open(corpus_filepath, 'w', encoding='utf-8')
	lines = []
	for lang_code in [lang1, lang2]:
		print_status(lang_code)
		lang_name = langs[lang_code]['name']
		try:
			if (lang_code == 'arz'): lang_code = 'ar'
			module = importlib.import_module("spacy.lang." + lang_code)
			nlp = getattr(module, lang_name)()
		except:
			nlp = spacy.language.Language()
		tokenizer = nlp.Defaults.create_tokenizer(nlp)

		filepath = dataset_filepath + lang_code
		print_status(filepath)
		file = open(filepath, 'rt', encoding='utf8')
		for l in file:
			if(l.strip() == ''): # skip empty lines
				continue
			# Clean XML tags
			cleantext = BeautifulSoup(l, "lxml").text 
			# Tokenize tweets
			tokens = list(tokenizer(cleantext)) 
			# Join tokens to create a single string for each line
			l = []
			for t in tokens:
				l.append(t.text)
			l = " ".join(l)
			# Add line in concatenated corpus
			lines.append(l)
		file.close()

	print("Sentences shuffled length: ")
	print(len(lines))
	# Shuffle English and Spanish tweets
	random.shuffle(lines)
	for l in lines:
		words = l.split()
		for w in words:
			w = w.lower()
			out.write(w + ' ')
		out.write("\n")

	print_status('Data saved at: ' + corpus_filepath)

# Get language code from keyboard
if len(sys.argv) == 1:
	print('Please enter language codes as arg (en, es, hi-ro, ne-ro, ar, arz).')
	exit(1)

if ['en', 'es','hi-ro', 'ne-ro', 'ar', 'arz'].count(sys.argv[1]) == 0 or ['en', 'es', 'hi-ro', 'ne-ro', 'ar', 'arz'].count(sys.argv[2]) == 0:
	print('Language code should be either en, es, hi-ro, ne-ro, ar or arz')
	exit(1)

# Language codes
langs = {
	'en': {
		'code': 'en',
		'name': 'English'
	},
	'es': {
		'code': 'es',
		'name': 'Spanish'
	},
	'hi-ro': {
		'code': 'hi-ro',
		'name': 'Hindi-Romanized'
	},
	'ne-ro': {
		'code': 'ne-ro',
		'name': 'Nepali-Romanized'
	},
	'ar': {
		'code': 'ar',
		'name': 'Arabic'
	},
	'arz': {
		'code': 'arz',
		'name': 'Arabic' # Egyptian Arabic
	},
}

lang1 = sys.argv[1]
lang2 = sys.argv[2]

# Fasttext embeddings from bilingual training data
corpus_filepath = './datasets/bilingual-annotated/' + lang1 + '-' + lang2 + '/train.conll' # train
embeddings_filepath = './embeddings/embeddings_fasttext_bilingual_' + lang1 + '_' + lang2 + '.bin'
_, _, sentences = read_lince_labeled_data(corpus_filepath)
save_train_data(sentences, './train.txt')
model_fasttext = fasttext.train_unsupervised('./train.txt', dim=100)
model_fasttext.save_model(embeddings_filepath)
os.remove('./train.txt')

# Fasttext embeddings from concatenated articles
"""
corpus_filepath = './datasets/concatenated_' + lang1 + '_' + lang2 + '.txt'
embeddings_filepath = './embeddings/embeddings_fasttext_concatenated_shuffled_' + lang1 + '_' + lang2 + '.bin'
create_corpus_articles(lang1, lang2, langs, embeddings_filepath, corpus_filepath)
model_fasttext = fasttext.train_unsupervised(corpus_filepath, dim=100)
model_fasttext.save_model(embeddings_filepath)
"""

# Fasttext embeddings from concatenated tweets
"""
corpus_filepath = './datasets/tweets_concatenated_other_' + lang1 + '_' + lang2 + '.txt'
dataset_filepath = './datasets/tweets/tweets.'
embeddings_filepath = './embeddings/embeddings_fasttext_tweets_concatenated_shuffled_' + lang1 + '_' + lang2 + '.bin'
create_corpus_tweets(lang1, lang2, langs, corpus_filepath, dataset_filepath)
model_fasttext = fasttext.train_unsupervised(corpus_filepath, dim=100)
model_fasttext.save_model(embeddings_filepath)
"""