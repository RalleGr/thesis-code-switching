#############################################################
# Source code for paper : A two-step training approach for semi-supervised language identification in code-switched data
# Authors: Dana-Maria Iliescu, Rasmus Grand & Sara Qirko
# IT-University of Copenhagen
# May 2021
#############################################################

from models.network.bilstm import BiLSTM
from models.network.lstm import LSTM
from models.network.simple_rnn import SimpleRNN
import sys
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from models.network.word import Word
from models.network.sentence import Sentence
from tools.utils import print_status
import json
import tensorflow as tf
import numpy as np
import argparse
from tensorflow import keras
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import Input
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tools.utils import read_lince_labeled_data
from tools.utils import is_other
from tools.utils import read_file
from tools.utils import failed_instances_to_csv
import fasttext
# Comment this line out for testing with BERT Embeddings
# Requires tensorflow==2.2.0 and latest transformers package (git+https://github.com/huggingface/transformers)
# from transformers import BertTokenizer, TFBertModel, AutoTokenizer, TFAutoModel

np.random.seed(1234)
tf.random.set_seed(1234)

# Constant Parameters
arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("lang1",type=str,help="Language code for language one, choose from [en,es,hi-ro,ne-ro,ar,arz]")
arg_parser.add_argument("lang2",type=str,help="Language code for language two, choose from [en,es,hi-ro,ne-ro,ar,arz]")

arg_parser.add_argument("architecture",type=str,help="Choose one architecture of [bilstm,lstm,simple_rnn]")
arg_parser.add_argument("units",type=int,help="Select a number of units")
arg_parser.add_argument("embedding",type=str,help="Choose one of the following embeddings [fasttext_bilingual,fasttext_concatenated_shuffled,fasttext_tweets_concatenated_shuffled,bert_wiki,bert_twitter]")
arg_parser.add_argument("optimizer",type=str,help="Choose one of the following optimizers [sgd,adam]")
arg_parser.add_argument("learning_rate",type=float,help="Choose a learning rate for the optimizer [range 0.0 to 1.0]")
arg_parser.add_argument("epochs",type=int,help="Select an amount of epochs [20,30,40,50,60,70]")
arg_parser.add_argument("batch_size",type=int,help="Choose a batch size [8,16,32,64]")

arg_parser.add_argument("--eta_lr",type=float,nargs='?',const=0.05,help="Choose a learning rate for eta [in paper code: 0.05]")
arg_parser.add_argument("--pretrained_model_name",type=str,help="Give the name of a pre-trained model for calculating label noise probabilities.")

arg_parser.add_argument("--momentum",type=float,help="Choose a momentum [range 0.0 to 1.0]")
arg_parser.add_argument("--gpu",dest='gpu',action='store_true',help="Use GPU hardware resources")
arg_parser.add_argument("--log",dest='log',action='store_true',help="Use Tensorflow logging callback for profiling")
arg_parser.add_argument("--mixed",dest='mixed',action='store_true',help="Use mixed precision e.g float16 when running on GPU")

args = arg_parser.parse_args()

lang1 = args.lang1
lang2 = args.lang2
architecture = args.architecture
units = args.units
embedding_type = args.embedding
optimizer = args.optimizer
learning_rate = args.learning_rate #5e-3 # α2
epochs = args.epochs
batch_size = args.batch_size
momentum = args.momentum

eta_lr = args.eta_lr # 5e-2 # α1
pretrained_model_name = args.pretrained_model_name

use_gpu = args.gpu
use_logging = args.log
use_mixed = args.mixed

if use_gpu:
	# Set GPU parameters if available
	gpus = tf.config.list_physical_devices('GPU')
	os.environ['CUDA_VISIBLE_DEVICES'] = '1'
	if gpus:
		try:
			# Currently, memory growth needs to be the same across GPUs
			for gpu in gpus:
				tf.config.experimental.set_memory_growth(gpu, True)
			logical_gpus = tf.config.experimental.list_logical_devices('GPU')
			print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
			if use_mixed:
				print_status("Enabled Mixed Precision")
				policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16', loss_scale='dynamic')
				tf.keras.mixed_precision.experimental.set_policy(policy)
		except RuntimeError as e:
			# Memory growth must be set before GPUs have been initialized
			print_status(e)
else:
	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
	return tf.convert_to_tensor(x, dtype='float32')

def extract_features(sentences):
	x_char_features = []
	x_word_features = []
	for s in sentences:
		x_char_features.append(s[0])
		x_word_features.append(s[1])
	return x_char_features, x_word_features

def extract_tokens(val_sentence):
	tokens = []
	labels = []
	for token_label in val_sentence:
		tokens.append(token_label[0])
		labels.append(token_label[1])
	return tokens, labels

ETA_init = None
def get_ETA_init():
	global ETA_init
	if ETA_init is None:
		ETA_init = tf.Variable(tf.zeros((x_char_train.shape[0],x_char_train.shape[1] + x_word_train.shape[1]), dtype='float32') + eta_init)
	return ETA_init


@tf.function
def training_loop(x_batch_train, t_batch_train, step, model, opt, train_acc_metric, ETA, psi, eta_lr):
	with tf.GradientTape() as tape:
		tape.watch(x_batch_train)
		# Run the forward pass of the layer.
		logits = model(x_batch_train, training=True) # Logits for this minibatch

		# 5: # predicting step
		# 6: q_i = 1/K_i h_w (x_i) * [(1 − η_i) y_hat_i + η_i * ψ_i * 1], i ∈ Ξ;
		q = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
		for i in tf.range(len(logits)):
			q_s = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
			for j in tf.range(len(logits[i])):
				eta_w = ETA[batch_size * step + i][j]
				res = logits[i][j] * tf.cast(((1-eta_w) * t_batch_train[i][j] + eta_w * psi[batch_size * step + i][j] * np.ones(shape=(3,), dtype='float32')), dtype='float32')
				q_w = res / tf.math.reduce_sum(res)
				# 7: # updating step
				# 8: if update posterior then
				if(epoch != 0):
					q_w = tf.reshape(q_w, [-1, 1])
					# 9
					eta_w = eta_w + eta_lr * ((tf.matmul(tf.reshape((np.ones(shape=(3,)) + (psi[batch_size * step + i][j] * eta_w - eta_w -1) * logits[i][j]), [1, -1]), q_w)) / (eta_w + 0.0001))
					# 10
					eta_new_value = tf.math.minimum(tf.math.maximum(eta_w[0][0], 0), 1)
					eta_row_length = len(ETA[batch_size * step + i])
					eta_row_new = tf.TensorArray(tf.float32, size=eta_row_length)
					for k in tf.range(eta_row_length):
						if (k == j):
							eta_row_new = eta_row_new.write(k, eta_new_value)
						else:
							eta_row_new = eta_row_new.write(k, ETA[batch_size * step + i][j])
				
					i_s = tf.IndexedSlices(tf.reshape(eta_row_new.stack(),[1, eta_row_length]), [batch_size * step + i])
					ETA.scatter_update(i_s)

				q_w = asTensor(tf.reshape(q_w, (3,)))
				q_s = q_s.write(j, q_w)
			
			q = q.write(i, q_s.stack())
		
		loss_value = tf.keras.backend.mean(tf.keras.losses.categorical_crossentropy(q.stack(), logits))

		# Use the gradient tape to automatically retrieve
		# the gradients of the trainable variables with respect to the loss.
		grads = tape.gradient(loss_value, model.trainable_weights)

		# Run one step of gradient descent by updating
		# the value of the variables to minimize the loss.
		opt.apply_gradients(zip(grads, model.trainable_weights))

		# Update training accuracy
		train_acc_metric.update_state(t_batch_train, logits)

# Load train data
print_status('Loading training data...')
zippath = 'training-data/training-data.zip'
filepath = 'training-data/network_train_' + embedding_type + '_' + lang1 + '_' + lang2 + '.json'
with read_file(filepath,zippath) as train_data_f:
	encoded_sentences = json.load(train_data_f)
	train_data_f.close()

x_sentences = []
t_sentences = []
s_max_length = -1

# Read training data
for s in encoded_sentences:
	s = Sentence(encoded_sentence=s)
	s_vectors_only = []
	t_s = []
	char_s = []
	if (len(s.words) > s_max_length): s_max_length = len(s.words)
	for w in s.words:
		if (ord(w.word[0]) == 65039): continue
		if (w.label != 'other'):
			char_s.append(get_tokenizer_char_features(w.word))
			s_vectors_only.append(w.vector)
			if (w.label == 'lang1'): t_s.append(1)
			if (w.label == 'lang2'): t_s.append(2)
	x_sentences.append((char_s, s_vectors_only))
	t_sentences.append(t_s)

nr_batches = int(np.floor(len(x_sentences) / batch_size))
x_sentences = x_sentences[:nr_batches * batch_size]
t_sentences = t_sentences[:nr_batches * batch_size]

# Extract char and word features for training and validation
x_char_train, x_word_train = extract_features(x_sentences)

# Pad sequences for training
x_char_train = sequence.pad_sequences(x_char_train, maxlen=s_max_length, dtype='float32')
x_word_train = sequence.pad_sequences(x_word_train, maxlen=s_max_length, dtype='float32')
t_train = sequence.pad_sequences(t_sentences, maxlen=s_max_length + x_char_train.shape[1], dtype='float32')

# One-hot encode T
t_train = to_categorical(t_train)

# Convert everything to tensors
x_char_train, x_word_train = asTensor(x_char_train), asTensor(x_word_train)
t_train = asTensor(t_train)

train_dataset = tf.data.Dataset.from_tensor_slices(((x_char_train, x_word_train), t_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

print("-"*100)
print("Sequence length (max): ", s_max_length) # 83
print("X char train shape: ", x_char_train.shape) # (16793, 22, 40)
print("X word train shape: ", x_word_train.shape) # (16793, 83, 100)
print("T train shape: ", t_train.shape) # (16793, 83)

print("-"*100)

# Input: the noisy-labeled training sample (S, Ye);
# 		the estimated probabilities {ψ1, . . . , ψN };
# 		number of training steps num_steps;
# 		hyperparameters η_INIT, α1, α2.
input_char_shape = [x_char_train.shape[1], x_char_train.shape[2]]
input_word_shape = [x_word_train.shape[1], x_word_train.shape[2]]

pretrained_model_filepath = 'models/train_models/' + pretrained_model_name
pretrained_model = keras.models.load_model(pretrained_model_filepath)
estimated_probabilities_psi = pretrained_model.predict([x_char_train, x_word_train])
eta_init = 0.01 # η_INIT
train_count = x_char_train.shape[0]

model_filepath = 'models/train_models/model_idn' + lang1 + '_' + lang2 + '_' + architecture + '_' + str(units) + '_' + embedding_type + '_' + \
				optimizer + '_lr' + str(learning_rate) + '_ep' + str(epochs) + '_bsize' + str(batch_size) + '_mom' + str(momentum) + '_eta_lr' + str(eta_lr) + '.h5'
if(not os.path.exists(model_filepath)):
	# Get the highest estimated probability for each word in the sentences
	psi = []
	for s in estimated_probabilities_psi:
		psi_s = []
		for w in s:
			psi_s.append(max(w))
		psi.append(psi_s)
	psi = asTensor(psi)

	# 1: Initialize η_i = η_INIT, ∀i;
	ETA = get_ETA_init()

	# 2: Initialize parameters w;
	if(architecture == 'bilstm'):
		model, opt = BiLSTM(input_char_shape,input_word_shape,optimizer=optimizer,embedding_type=embedding_type,main_output_lstm_units=units,learning_rate=learning_rate,momentum=momentum,return_optimizer=True)
	elif(architecture == 'lstm'):
		model, opt = LSTM(input_char_shape,input_word_shape,optimizer=optimizer,embedding_type=embedding_type,main_output_lstm_units=units,learning_rate=learning_rate,momentum=momentum,return_optimizer=True)
	elif(architecture == 'simple_rnn'):
		model, opt = SimpleRNN(input_char_shape,input_word_shape,optimizer=optimizer,embedding_type=embedding_type,main_output_lstm_units=units,learning_rate=learning_rate,momentum=momentum,return_optimizer=True)

	model.summary()

	train_acc_metric = keras.metrics.CategoricalAccuracy()
	
	# 3: for t ← 1 to num steps do
	for epoch in tf.range(epochs):
		print_status("Start of epoch " + str(epoch))

		# Iterate over the batches of the dataset.
		# 4: Fetch a mini-batch Ξ uniformly at random;
		for step, (x_batch_train, t_batch_train) in enumerate(train_dataset):
			step = tf.cast(step, dtype='int32')
			training_loop(x_batch_train, t_batch_train, step, model, opt, train_acc_metric, ETA, psi, eta_lr)
		
		# Display metrics at the end of each epoch
		train_acc = train_acc_metric.result().numpy()
		tf.print("Training acc over epoch: %.4f" % (float(train_acc),))

		# Reset training metrics at the end of each epoch
		train_acc_metric.reset_states()

	# Save model
	model.save(model_filepath)
else:
	model = keras.models.load_model(model_filepath)

# Evaluate the model
# Train accuracy
_, train_acc = model.evaluate([x_char_train, x_word_train], t_train, verbose=1)

# Load dev data
filepath = 'datasets/bilingual-annotated/' + lang1 + '-' + lang2 + '/dev.conll'
val_tokens, val_labels, val_sentences = read_lince_labeled_data(filepath)

# Load embeddings model
if ('fasttext' in embedding_type):
	embeddings_filepath = './embeddings/embeddings_' + embedding_type + '_' + lang1 + '_' + lang2 + '.bin'
	fasttext_model = fasttext.load_model(embeddings_filepath)
	print_status("Embeddings loaded")
else:
	if (embedding_type == 'bert_wiki'):
		tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
		bert_model = TFBertModel.from_pretrained('bert-base-multilingual-cased')
	elif (embedding_type == 'bert_twitter'):
		tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-xlm-roberta-base')
		bert_model = TFAutoModel.from_pretrained('cardiffnlp/twitter-xlm-roberta-base')

# Create lists for network evaluation (only lang1 and lang2) and overall evaluation (+ 'other' tokens)
x_char_test = []
x_word_test = []
t_test = []

for s in val_sentences:
	char_s = []
	word_s = []
	t_s = []
	if ('fasttext' in embedding_type):
		for i in range(len(s)):
			token = s[i][0]
			label = s[i][1]

			if (is_other(token) == False):
				token_char_vector = get_tokenizer_char_features(token)
				token_word_vector = fasttext_model.get_word_vector(token)
				char_s.append(token_char_vector)
				word_s.append(token_word_vector)
				t_s.append(1 if label == 'lang1' else 2)
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
			token = s[i][0]
			label = s[i][1]

			if (is_other(token) == False):
				token_char_vector = get_tokenizer_char_features(token)
				subwords_count = len(tokenizer.tokenize(token))
				token_word_vector = np.mean(embeddings[idx : idx + subwords_count], 0)
				char_s.append(token_char_vector)
				word_s.append(token_word_vector)
				t_s.append(1 if label == 'lang1' else 2)
				idx += subwords_count
	x_char_test.append(char_s)
	x_word_test.append(word_s)
	t_test.append(t_s)

# Pad sequences
x_char_test = sequence.pad_sequences(x_char_test, maxlen=s_max_length, dtype='float32')
x_word_test = sequence.pad_sequences(x_word_test, maxlen=s_max_length, dtype='float32')
t_test_network = sequence.pad_sequences(t_test, maxlen=s_max_length + x_char_train.shape[1], dtype='float32')

# One-hot encode T
t_test_network = to_categorical(t_test_network)

# Convert to tensor
x_char_test, x_word_test, t_test_network = asTensor(x_char_test), asTensor(x_word_test), asTensor(t_test_network)

_, val_acc = model.evaluate([x_char_test, x_word_test], t_test_network, verbose=1)
print('Embedding type: %s' % embedding_type)
print('Train accuracy (network): %.4f' % train_acc)
print('Validation accuracy (network): %.4f' % val_acc)

# Get predictions on validation data
model_predictions = model.predict([x_char_test, x_word_test])
y = []
token_sentences = []
label_sentences = []
for i in range(len(val_sentences)):
# for i in range(0, 10):
	if(len(val_sentences[i]) > 0):
		tokens, labels = extract_features(val_sentences[i])
		token_sentences.append(tokens)
		label_sentences.append(labels)
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

# Qualitative analysis
failed_instances_to_csv("idn-instances", token_sentences, y, label_sentences)

t = val_labels
y = [item for y_sent in y for item in y_sent] # Flatten list
val_acc_overall = accuracy_score(t, y)
f1_score_weighted = f1_score(t, y, average='weighted')
f1_score_class = f1_score(t, y, average=None)
print('Validation accuracy (overall): %.4f' % val_acc_overall)
print('Validation F1 score weighted (overall): %.4f' % f1_score_weighted)
print('Validation F1 score per class (overall): %s' % f1_score_class)
print_status('End')
