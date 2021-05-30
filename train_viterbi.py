#############################################################
# Source code for paper : A two-step training approach for semi-supervised language identification in code-switched data
# Authors: Dana-Maria Iliescu, Rasmus Grand & Sara Qirko
# IT-University of Copenhagen
# May 2021
#############################################################

from models.viterbi.viterbi_language_model import ViterbiLanguageModel
from bs4 import BeautifulSoup
from tools.utils import write_dict
from tools.utils import is_other
from tools.utils import print_status
from tools.utils import printProgressBar
import spacy
import os
import importlib
import sys

WORD_DICTIONARIES_PATH = "./dictionaries/word-level/"
CHAR_DICTIONARIES_PATH = "./dictionaries/char-level/"

def get_frequency_dict(lang_code, lang_name):
	print_status("Creating frequency dictionaries...")

	frequency_dict = dict()

	# Load data
	for root, dirs, files in os.walk('datasets/monolingual-' + lang_code):
		if ('.DS_Store' in files):
			files.remove('.DS_Store')
		for f in files:
			print_status(f"File : {f}")
			filepath = os.path.join(root, f)
			file = open(filepath, 'rt', encoding='utf8')
			text = file.read()
			file.close()
			print_status(f"Cleaning XML and HTML")
			# Clean XML tags
			cleantext = BeautifulSoup(text, "lxml").text

			print_status("Tokenizing")
			try:
				if (lang_code == 'arz'): lang_code = 'ar'
				module = importlib.import_module("spacy.lang." + lang_code)
				nlp = getattr(module, lang_name)()
			except:
				nlp = spacy.language.Language()
			tokenizer = nlp.Defaults.create_tokenizer(nlp)
			tokens = list(tokenizer(cleantext))

			i = 0
			printProgressBar(0, len(tokens), prefix = 'Progress:', suffix = 'Total tokens', length = 50)
			for word in tokens:
				word = word.text.lower()

				if is_other(word):
					continue
				else:
					if word in frequency_dict.keys():
						frequency_dict[word] += 1
					else:
						frequency_dict[word] = 1

				printProgressBar(i + 1, len(tokens), prefix = 'Progress:', suffix = 'Total tokens', length = 50)
				i+=1

	return frequency_dict

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

# Get language code from keyboard
if len(sys.argv) == 1:
	print("Please enter language code as arg (en, es, hi-ro, ne-ro, ar, arz).")
	exit(1)

if ['en', 'es', 'hi-ro', 'ne-ro', 'ar', 'arz'].count(sys.argv[1]) == 0:
	print("Language code should be either en, es, hi-ro, ne-ro, ar, arz")
	exit(1)
lang = sys.argv[1]
lang_code = langs[lang]['code']
lang_name = langs[lang]['name']

# n value for ngrams
ngram = 2

# Unigram frequency lexicons.
lex_path = WORD_DICTIONARIES_PATH + 'frequency_dict_' + lang_code + '.csv'
if (not os.path.exists(lex_path)):
	frequency_dict = get_frequency_dict(lang_code, lang_name)
	write_dict(WORD_DICTIONARIES_PATH, frequency_dict, 'frequency_dict_' + lang_code)

# Create models
lang_model = ViterbiLanguageModel(lang, ngram, lex_path)

# Train and save character n-gram models.
lang_model.train()
lang_model_path = CHAR_DICTIONARIES_PATH + str(ngram) + '-gram-' + lang_code + '.lm'
lang_model.dump(lang_model_path)

print_status("Done!")