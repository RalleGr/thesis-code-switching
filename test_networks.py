#############################################################
# Source code for paper : A two-step training approach for semi-supervised language identification in code-switched data
# Authors: Dana-Maria Iliescu, Rasmus Grand & Sara Qirko
# IT-University of Copenhagen
# May 2021
#############################################################
from tools.utils import read_lince_unlabeled_data
from tools.utils import read_lince_labeled_data
from tools.utils import save_predictions
from tools.utils import is_other
from tools.utils import print_status
from models.network.bilstm import BiLSTM
import numpy as np
import sys
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
import fasttext
# Comment this line out for testing with BERT Embeddings
# Requires tensorflow==2.2.0 and latest transformers package (git+https://github.com/huggingface/transformers)
# from transformers import BertTokenizer, TFBertModel, AutoTokenizer, TFAutoModel

def get_tokenizer_char_features(word,max_length=20):
	word = word[:max_length]
	word = '<' + word + '>'
	alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ñ', 'á', 'é', 'í', 'ó', 'ú', 'ü', '#', '\'','’', '<', '>','UNK']
	tokenizer = Tokenizer(char_level=True, filters='!"$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', oov_token='UNK')
	tokenizer.fit_on_texts(alphabet)
	sequence_of_integers = tokenizer.texts_to_sequences(word)
	flat_sequence = np.array(sequence_of_integers).flatten().tolist()
	while (len(flat_sequence) < max_length + 2):
		flat_sequence.append(-1)
	return to_categorical(flat_sequence).flatten()

def asTensor(x):
	return tf.convert_to_tensor(x)

def extract_features(sentences):
	x_char_features = []
	x_word_features = []
	for s in sentences:
		x_char_features.append(s[0])
		x_word_features.append(s[1])
	return x_char_features, x_word_features

lang1 = sys.argv[1]
lang2 = sys.argv[2]
dataset = sys.argv[3]
embedding_type = sys.argv[4]
model_name = sys.argv[5]
use_gpu = True

if use_gpu:
	# Set GPU parameters if available
	gpus = tf.config.list_physical_devices('GPU')
	if gpus:
		try:
			# Currently, memory growth needs to be the same across GPUs
			for gpu in gpus:
				tf.config.experimental.set_memory_growth(gpu, True)
			logical_gpus = tf.config.experimental.list_logical_devices('GPU')
			print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
		except RuntimeError as e:
			# Memory growth must be set before GPUs have been initialized
			print_status(e)
else:
	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if (lang2 == 'es'):
	s_max_length = 83
	if ('incv' in model_name):
		s_max_length = 34
elif (lang2 == 'hi-ro'):
	s_max_length = 247
elif (lang2 == 'ne-ro'):
	s_max_length = 39
elif (lang2 == 'arz'):
	s_max_length = 37
	if ('incv' in model_name):
		s_max_length = 32

# Load trained model
model_filepath = 'models/train_models/' + model_name
try:
	model = keras.models.load_model(model_filepath)
except:
	if (lang2 == 'hi-ro' or lang2 == 'ne-ro'):
		input_char_shape = [s_max_length, 880]
		input_word_shape = [s_max_length, 100]
		model = BiLSTM(input_char_shape,input_word_shape,optimizer='adam',embedding_type='fasttext_bilingual',main_output_lstm_units=64,learning_rate=0.01,momentum=0.9,return_optimizer=False)
		model.load_weights(model_filepath)
print_status("Trained model loaded")

# Load test data
if (dataset == 'test'):
	filepath = 'datasets/bilingual-annotated/' + lang1 + '-' + lang2 + '/test.conll'
	_, test_sentences = read_lince_unlabeled_data(filepath)
elif (dataset == 'dev'):
	filepath = 'datasets/bilingual-annotated/' + lang1 + '-' + lang2 + '/dev.conll'
	_, _, test_sentences_tuples = read_lince_labeled_data(filepath)
	test_sentences = []
	for i in range(0, len(test_sentences_tuples)):
		test_s = []
		if (len(test_sentences_tuples[i]) > 0):
			for w, l in test_sentences_tuples[i]:
				test_s.append(w)
		test_sentences.append(test_s)
print_status("Test data loaded")

# Load embeddings model
if ('fasttext' in embedding_type):
	embeddings_filepath = './embeddings/embeddings_' + embedding_type + '_' + lang1 + '_' + lang2 + '.bin'
	fasttext_model = fasttext.load_model(embeddings_filepath)
else:
	if (embedding_type == 'bert_wiki'):
		tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
		bert_model = TFBertModel.from_pretrained('bert-base-multilingual-cased')
	elif (embedding_type == 'bert_twitter'):
		tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-xlm-roberta-base')
		bert_model = TFAutoModel.from_pretrained('cardiffnlp/twitter-xlm-roberta-base')
print_status("Embeddings loaded")

# Create lists for network evaluation (only lang1 and lang2) and overall evaluation (+ 'other' tokens)
x_char_test = []
x_word_test = []

for s in test_sentences:
	char_s = []
	word_s = []
	if ('fasttext' in embedding_type):
		for i in range(len(s)):
			token = s[i]

			if (is_other(token) == False):
				token_char_vector = get_tokenizer_char_features(token)
				token_word_vector = fasttext_model.get_word_vector(token)
				char_s.append(token_char_vector)
				word_s.append(token_word_vector)
		x_char_test.append(char_s)
		x_word_test.append(word_s)
	else:
		# Add the special tokens.
		marked_text = "[CLS] " + ' '.join([tup[0] if is_other(tup[0]) == False else '' for tup in s]) + " [SEP]"

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
		for i in range(len(s)):
			token = s[i]

			if (is_other(token) == False):
				token_char_vector = get_tokenizer_char_features(token)
				subwords_count = len(tokenizer.tokenize(token))
				token_word_vector = np.mean(embeddings[idx : idx + subwords_count], 0)
				char_s.append(token_char_vector)
				word_s.append(token_word_vector)
				idx += subwords_count

# Pad sequences
x_char_test = sequence.pad_sequences(x_char_test, maxlen=s_max_length, dtype='float32')
x_word_test = sequence.pad_sequences(x_word_test, maxlen=s_max_length, dtype='float32')

# Convert to tensor
x_char_test, x_word_test = asTensor(x_char_test), asTensor(x_word_test)

print("X char test shape: ", x_char_test.shape)
print("X word test shape: ", x_word_test.shape)

# Get predictions on test data
print_status("Predicting...")
model_predictions = model.predict([x_char_test, x_word_test])
y = []
for i in range(len(test_sentences)):
	if(len(test_sentences[i]) > 0):
		tokens = test_sentences[i]
		# Separate 'lang' words from 'other' words
		lang_tokens = []
		other_indexes = []
		for j in range(len(tokens)):
			if (is_other(tokens[j])): other_indexes.append(j)
			else:
				lang_tokens.append(tokens[j])
		
		# For sentences with 'lang1', 'lang2' and 'other' words
		if(len(lang_tokens) > 0):
			y_sentence = model_predictions[i]
			y_sentence = y_sentence[-len(lang_tokens):]
			y_sentence = ['lang1' if y_token[1] > y_token[2] else 'lang2' for y_token in y_sentence]

			# Insert 'other' label back into the sentence at the right indexes
			for index in other_indexes:
				y_sentence.insert(index, 'other')

		# For sentences that are made up only of 'other' words
		else:
			y_sentence = []
			for index in other_indexes:
				y_sentence.append('other')

		y.append(y_sentence)

if (dataset == 'dev'):
	y_index = 0
	for i in range(0, len(test_sentences)):
		if (len(test_sentences[i]) > 0):
			for j in range(0, len(test_sentences[i])):
				y[y_index][j] = (test_sentences[i][j], y[y_index][j]) # for dev, save token + prediction. mixed, ne, etc. classes are skipped
			y_index += 1

predictions_filepath= './predictions/' + dataset + '/' + lang1 + '-' + lang2 
if not os.path.exists("./predictions/"):
	os.mkdir("./predictions/")
if not os.path.exists(f"./predictions/{dataset}"):
	os.mkdir(f"./predictions/{dataset}")
if not os.path.exists(f"./predictions/{dataset}/{lang1}-{lang2}"):
	os.mkdir(f"./predictions/{dataset}/{lang1}-{lang2}")

save_predictions(y, predictions_filepath + '/' + model_name + '.txt')