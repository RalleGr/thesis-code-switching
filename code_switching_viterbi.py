#############################################################
# Source code for paper : A two-step training approach for semi-supervised language identification in code-switched data
# Authors: Dana-Maria Iliescu, Rasmus Grand & Sara Qirko
# IT-University of Copenhagen
# May 2021
#############################################################

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from models.viterbi.viterbi_identifier import ViterbiIdentifier
from tools.utils import is_other
from tools.utils import print_status
from tools.utils import save_predictions
from tools.utils import save_train_data
from tools.utils import read_lince_labeled_data
from tools.utils import read_lince_unlabeled_data
from tools.utils import failed_instances_to_csv
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from tools.logger import Logger
import matplotlib.pyplot as plt
import sys
from models.network.word import Word
from models.network.sentence import Sentence
import fasttext
import json
import numpy as np
import tensorflow as tf
# Comment this line out for testing with BERT Embeddings
# Requires tensorflow==2.2.0 and latest transformers package (git+https://github.com/huggingface/transformers)
# from transformers import BertTokenizer, TFBertModel, AutoTokenizer, TFAutoModel

logger = Logger("viterbi")
#logger.begin()

# This version was used as final model

WORD_DICTIONARIES_PATH = "./dictionaries/word-level/"
CHAR_DICTIONARIES_PATH = "./dictionaries/char-level/"

# Get language codes from keyboard
if len(sys.argv) != 5:
	print("Please enter: lang1, lang2, evaluation dataset and embedding type.")
	print("Language codes: choose between en es, en hi-ro, en ne-ro or ar arz")
	print("Evaluation dataset: choose between train, dev, test, train_dev (for other language pair than en es)")
	print("Embedding type: fasttext_bilingual,fasttext_concatenated_shuffled,fasttext_tweets_concatenated_shuffled,bert_wiki,bert_twitter")
	exit(1)

lang1 = sys.argv[1]
lang2 = sys.argv[2]
evaluation_dataset = sys.argv[3]
embedding_type = sys.argv[4]

# Language model files.
print_status("Loading model...")
ngram = 2
lang1_lm = CHAR_DICTIONARIES_PATH + str(ngram) + '-gram-' + lang1 + '.lm'
lang2_lm = CHAR_DICTIONARIES_PATH + str(ngram) + '-gram-' + lang2 + '.lm'

# Unigram frequency lexicons.
lang1_lex = WORD_DICTIONARIES_PATH + 'frequency_dict_' + lang1 + '.csv'
lang2_lex = WORD_DICTIONARIES_PATH + 'frequency_dict_' + lang2 + '.csv'

identifier = ViterbiIdentifier(lang1_lm, lang2_lm,
								lang1_lex, lang2_lex)

# Get data
print_status("Getting test data...")
print_status("Language pair: " + lang1 + ' - ' + lang2)

if (evaluation_dataset == 'train'):
	filepath = './datasets/bilingual-annotated/' + lang1 + '-' + lang2 + '/train.conll' # train
if (evaluation_dataset == 'dev'):
	filepath = './datasets/bilingual-annotated/' + lang1 + '-' + lang2 + '/dev.conll' # validation
if (evaluation_dataset == 'test'):
	filepath = './datasets/bilingual-annotated/' + lang1 + '-' + lang2 + '/test.conll' # test
if (evaluation_dataset == 'train_dev'):
	filepath = './datasets/bilingual-annotated/' + lang1 + '-' + lang2 + '/train_dev.conll' # train + validation for other language pairs

file = open(filepath, 'rt', encoding='utf8')
sentences = []
t = []
s = []

if (evaluation_dataset != 'test'):
	_, t, sentences = read_lince_labeled_data(filepath)
else:
	_, sentences = read_lince_unlabeled_data(filepath)

y = []
predictions_dict = dict()
training_sentences = []
token_sentences = []
label_sentences = []
if (evaluation_dataset == 'test'):
	for tokens in sentences:
		if(len(tokens) > 0):
			# Separate 'lang' words from 'other' words
			lang_tokens = []
			other_indexes = []
			for i in range(len(tokens)):
				if (is_other(tokens[i])): other_indexes.append(i)
				else: lang_tokens.append(tokens[i])
			
			# For sentences with 'lang1', 'lang2' and 'other' words
			if(len(lang_tokens) > 0):
				y_sentence, conf_score, tokens_conf_score = identifier.identify(lang_tokens)
				for index in other_indexes:
					y_sentence.insert(index, 'other')

			# For sentences that are made up only of 'other' words
			else:
				y_sentence = []
				for index in other_indexes:
					y_sentence.append('other')
			y.append(y_sentence)
else:
	for tokens in sentences:
		if(len(tokens) > 0):
			words = []
			labels = []

			# Separate 'lang' words from 'other' words
			lang_tokens = []
			other_indexes = []
			for i in range(len(tokens)):
				words.append(tokens[i][0])
				labels.append(tokens[i][1])

				if (is_other(tokens[i][0])): other_indexes.append(i)
				else: lang_tokens.append(tokens[i][0])
			
			# For sentences with 'lang1', 'lang2' and 'other' words
			if(len(lang_tokens) > 0):
				y_sentence, conf_score, tokens_conf_score = identifier.identify(lang_tokens)
				for index in other_indexes: # Insert 'other' label and confidence score back into the sentence at the right indexes
					y_sentence.insert(index, 'other')
					tokens_conf_score.insert(index, -999)

			# For sentences that are made up only of 'other' words
			else:
				conf_score = -999
				y_sentence = []
				tokens_conf_score = []
				for index in other_indexes:
					y_sentence.append('other')
					tokens_conf_score.append(conf_score)
					
			y.append(y_sentence)
			token_sentences.append(words)
			label_sentences.append(labels)

			# For saving predictions to file
			for i in range(len(tokens)):
				predictions_dict[tokens[i][0]] = y_sentence[i]
			
			# For network training
			words = []
			for i in range(len(y_sentence)):
				w = Word(tokens[i][0], y_sentence[i], tokens_conf_score[i])
				words.append(w)
			training_s = Sentence(words, conf_score)
			training_sentences.append(training_s)

if (evaluation_dataset == 'test'):
	save_predictions(y, './results/predictions/predictions_test_original_viterbi-' + lang1 + '-' + lang2 + '.txt')
	print_status("Done!")
	exit(1)

if (evaluation_dataset == 'train' or evaluation_dataset == 'train_dev'):
	embeddings_filepath = './embeddings/embeddings_' + embedding_type + '_' + lang1 + '_' + lang2 + '.bin'
	if ('fasttext' in embedding_type):
		model = fasttext.load_model(embeddings_filepath)
		print_status("Embeddings loaded")
	else:
		if (embedding_type == 'bert_wiki'):
			tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
			bert_model = TFBertModel.from_pretrained('bert-base-multilingual-cased')
		elif (embedding_type == 'bert_twitter'):
			tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-xlm-roberta-base')
			bert_model = TFAutoModel.from_pretrained('cardiffnlp/twitter-xlm-roberta-base')

	i = 0
	for j in range(len(training_sentences)):
		s = training_sentences[j]
		s.set_id(j)
		if ('fasttext' in embedding_type):
			for w in s.words:
				word_vector = model.get_word_vector(w.word)
				w.set_vector(word_vector)
				w.set_id(i)
				i += 1
		else:
			# Add the special tokens.
			marked_text = "[CLS] " + ' '.join([w.word if w.label != 'other' else '' for w in s.words]) + " [SEP]"

			# Split the sentence into tokens.
			tokenized_text = tokenizer.tokenize(marked_text)

			# Map the token strings to their vocabulary indeces.
			indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

			# Mark each of the sentence tokens as belonging to sentence "1".
			segments_ids = [1] * len(tokenized_text)

			# Convert inputs to Tensorflow tensors
			indexed_tokens = tf.convert_to_tensor([indexed_tokens])
			segments_ids = tf.convert_to_tensor([segments_ids])

			# Run the text through BERT, and collect all of the hidden states produced
			# from all 12 layers.
			inputs = {'input_ids': indexed_tokens,
					'token_type_ids': segments_ids}

			if (embedding_type == 'bert_wiki'):
				outputs = bert_model(inputs)
			elif (embedding_type == 'bert_twitter'):
				outputs = bert_model(indexed_tokens)

			last_hidden_state = outputs[0] # shape = (nr_of_sentences, nr_of_tokens_per_sent, 768)
			embeddings = last_hidden_state[0]

			idx = 1
			for w in s.words:
				if w.label != 'other':
					subwords_count = len(tokenizer.tokenize(w.word))
					word_vector = np.mean(embeddings[idx : idx + subwords_count], 0)
					idx += subwords_count
					w.set_vector(word_vector)
				w.set_id(i)
				i += 1

	with open('./training-data/network_train_' + embedding_type + '_' + lang1 + '_' + lang2 + '.json', 'w', encoding='utf-8') as f:
		encoded_sentences = [s.json_encode() for s in training_sentences]
		json.dump(encoded_sentences, f)
	print_status("Done!")
	exit(1)

# Qualitative analysis
failed_instances_to_csv("viterbi-instances", token_sentences, y, label_sentences)

# Own test set with labels
# Flatten y list
y = [item for y_sent in y for item in y_sent]

# Get accuracy
acc = accuracy_score(t, y)
print("Accuracy: " + str(acc))

# F1 score
f1 = f1_score(t, y, average=None)
print("F1 score per class: " + str(f1))

# F1 score weighted
f1_weighted = f1_score(t, y, average='weighted')
print("Weighted F1 score: " + str(f1_weighted))

# Confusion matrix
conf_matrix = confusion_matrix(t, y)
classes = ['lang1', 'lang2', 'other']
ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes).plot(values_format='d')
plt.savefig('./results/CM/confusion_matrix_' + 'viterbi_v1.svg', format='svg')

# Save model output
if (evaluation_dataset == 'dev'):
	save_predictions(predictions_dict, './results/predictions/predictions_dev_viterbi-' + lang1 + '-' + lang2 + '.txt')
if (evaluation_dataset == 'test'):
	save_predictions(predictions_dict, './results/predictions/predictions_test_viterbi-' + lang1 + '-' + lang2 + '.txt')
print_status("Done!")
#logger.end()
