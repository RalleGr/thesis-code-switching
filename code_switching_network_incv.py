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
import gc
from models.network.sentence import Sentence
import json
import argparse
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
from tensorflow.keras import optimizers
from tensorflow.keras import Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from sklearn.model_selection import train_test_split
from tools.utils import read_lince_labeled_data
from tools.utils import is_other
from tools.utils import join
from tools.utils import subtract
from tools.utils import print_status
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

arg_parser.add_argument("--momentum",type=float,help="Choose a momentum [range 0.0 to 1.0]")
arg_parser.add_argument("--gpu",dest='gpu',action='store_true',help="Use GPU hardware resources")
arg_parser.add_argument("--log",dest='log',action='store_true',help="Use Tensorflow logging callback for profiling")
arg_parser.add_argument("--mixed",dest='mixed',action='store_true',help="Use mixed precision e.g float16 when running on GPU")

arg_parser.add_argument("--incv_epochs",type=int,help="Choose a number of INCV epochs [range 1 to 100]")
arg_parser.add_argument("--incv_iter",type=float,help="Choose a number of INCV iterations [range 1 to 4]")
arg_parser.add_argument("--remove_ratio",type=float,help="Choose a remove ratio [range 0.0 to 1.0]")

args = arg_parser.parse_args()

lang1 = args.lang1
lang2 = args.lang2
architecture = args.architecture
units = args.units
embedding_type = args.embedding
optimizer = args.optimizer
learning_rate = args.learning_rate
epochs = args.epochs
batch_size = args.batch_size
momentum = args.momentum

incv_epochs = args.incv_epochs
incv_iterations = args.incv_iter
remove_ratio = args.remove_ratio

use_gpu = args.gpu
use_logging = args.log
use_mixed = args.mixed

float_type = 'float32'

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
			if use_mixed:
				print_status("Enabled Mixed Precision")
				policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16', loss_scale='dynamic')
				tf.keras.mixed_precision.experimental.set_policy(policy)
		except RuntimeError as e:
			# Memory growth must be set before GPUs have been initialized
			print_status(e)
else:
	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def INCV_lr_schedule(epoch):
	# Learning Rate Schedule
	lr = 1e-3
	if epoch > 40:
		lr *= 0.1
	elif epoch > 30:
		lr *= 0.25
	elif epoch > 20:
		lr *= 0.5
	print_status('Learning rate: ' + str(lr))
	return lr

def get_tokenizer_char_features(word, max_length=20):
	word = word[:max_length]
	word = '<' + word + '>'
	alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ñ', 'á', 'é', 'í', 'ó', 'ú', 'ü', '#', '\'','’', '<', '>','UNK']
	tokenizer = Tokenizer(char_level=True, filters='!"$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', oov_token='UNK')
	tokenizer.fit_on_texts(alphabet)
	sequence_of_integers = tokenizer.texts_to_sequences(word)
	flat_sequence = np.array(sequence_of_integers,dtype=float_type).flatten().tolist()
	while (len(flat_sequence) < max_length + 2):
		flat_sequence.append(-1)
	return keras.utils.to_categorical(flat_sequence,dtype=float_type).flatten()

def asTensor(x):
	return tf.convert_to_tensor(x, dtype=float_type)

def extract_tokens(val_sentence):
	tokens = []
	labels = []
	for token_label in val_sentence:
		tokens.append(token_label[0])
		labels.append(token_label[1])
	return tokens, labels

def extract_features(sentences):
	ids = []
	x_char_features = []
	x_word_features = []
	t = []
	s_lengths = []
	for s in sentences:
		ids.append(s[0])
		x_char_features.append(s[1])
		x_word_features.append(s[2])
		t.append(s[3])
		s_lengths.append(s[4])
	return ids, x_char_features, x_word_features, t, s_lengths

@tf.function
def cross_entropy(a,b):
	if tf.keras.backend.floatx() == 'float32':
		tf.keras.backend.set_floatx(float_type)
	return tf.cast(tf.keras.backend.mean(tf.keras.losses.categorical_crossentropy(a,b)),dtype=float_type)

def fit_model(model, x_char_features, x_word_features, t, main_training=False):
	# Fit model
	print_status('Training model...')

	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1)

	callbacks = [ModelCheckpoint(filepath='{epoch:02d}.h5', monitor='train_acc', mode='max', verbose=1, save_best_only=True)]
	if (main_training): callbacks.append(EarlyStopping(monitor='train_loss', mode='min', verbose=1, patience=10))
	elif (use_logging): callbacks.append(tensorboard_callback)
	else: callbacks.append(LearningRateScheduler(INCV_lr_schedule))
	
	if (main_training):
		history = model.fit(
			[x_char_features, x_word_features],
			t,
			batch_size=batch_size,
			epochs=epochs,
			verbose=2,
			callbacks=callbacks
		)
	else:
		history = model.fit(
			[x_char_features, x_word_features],
			t,
			batch_size=batch_size,
			epochs=incv_epochs,
			verbose=2,
			callbacks=callbacks
		)
	print_status('Training done!')


def INCV(sentences, s_max_length):
	# 1: Selected set S = ∅, candidate set C = D
	S = []
	C = sentences

	# 3: Initialize Network
	ids, char_features, word_features, t, _ = extract_features(C)
	char_features = asTensor(sequence.pad_sequences(char_features, maxlen=s_max_length, dtype=float_type))
	word_features = asTensor(sequence.pad_sequences(word_features, maxlen=s_max_length, dtype=float_type))
	t = asTensor(to_categorical(sequence.pad_sequences(t, maxlen=s_max_length + char_features.shape[1],dtype=float_type)))

	input_char_shape = [char_features.shape[1], char_features.shape[2]]
	input_word_shape = [word_features.shape[1], word_features.shape[2]]
	if(architecture == 'bilstm'):
		model = BiLSTM(input_char_shape,input_word_shape,optimizer=optimizer,embedding_type=embedding_type,main_output_lstm_units=units,learning_rate=learning_rate,momentum=momentum,return_optimizer=False)
	elif(architecture == 'lstm'):
		model = LSTM(input_char_shape,input_word_shape,optimizer=optimizer,embedding_type=embedding_type,main_output_lstm_units=units,learning_rate=learning_rate,momentum=momentum,return_optimizer=False)
	elif(architecture == 'simple_rnn'):
		model = SimpleRNN(input_char_shape,input_word_shape,optimizer=optimizer,embedding_type=embedding_type,main_output_lstm_units=units,learning_rate=learning_rate,momentum=momentum,return_optimizer=False)

	model.summary()
	initial_weights = model.get_weights()
	eval_ratio = 0
	discard_ratio = None
	for incv_i in tf.range(incv_iterations):
		gc.collect()
		# 4: Randomly divide C into halved C1 and C2
		np.random.shuffle(C)
		C1, C2 = C[:len(C)//2], C[len(C)//2:]
		print_status('iteration: ' + str(incv_i) + ', finished step: 4')

		# 5: Train f(x; ω) on S ∪ C1 for E epochs
		joined_set = join(S,C1)
		print_status('iteration: ' + str(incv_i) + ', finished step: 5 (joining)')


		ids_C1, char_features_C1, word_features_C1, t_C1, _ = extract_features(joined_set)
		char_features_C1 = asTensor(sequence.pad_sequences(char_features_C1, maxlen=s_max_length, dtype=float_type))
		word_features_C1 = asTensor(sequence.pad_sequences(word_features_C1, maxlen=s_max_length, dtype=float_type))
		t_C1 = asTensor(to_categorical(sequence.pad_sequences(t_C1, maxlen=s_max_length + char_features_C1.shape[1],dtype=float_type),dtype=float_type))
		
		model.set_weights(initial_weights) # reset weights
		fit_model(model, char_features_C1, word_features_C1, t_C1, main_training=False)
		print_status('iteration: ' + str(incv_i) + ', finished step: 5 (training)')

		
		# 6: Select samples, S1 = {(x, y) ∈ C2 : y^f = y}
		S1 = []
		ids_C2, char_features_C2, word_features_C2, t_C2, s_lengths_C2 = extract_features(C2)
		char_features_C2 = asTensor(np.array(sequence.pad_sequences(char_features_C2, maxlen=s_max_length, dtype=float_type)))
		word_features_C2 = asTensor(np.array(sequence.pad_sequences(word_features_C2, maxlen=s_max_length, dtype=float_type)))

		y_C2 = model.predict([char_features_C2, word_features_C2])
		for i in tf.range(len(y_C2)):
			if (s_lengths_C2[i] == 0): continue
			s = y_C2[i][-s_lengths_C2[i]:] # [[0.2, 0.6, 0.2]... 83 times]
			same = True
			for j in tf.range(len(s)):
				token_prob = s[j]
				predicted_label = 1 if token_prob[1] > token_prob[2] else 2
				label = 1 if t_C2[i][j] == 1 else 2
				if (predicted_label != label):
					same = False
					break
			if (same):
				S1.append((ids_C2[i],char_features_C2[i],word_features_C2[i],t_C2[i],s_lengths_C2[i]))

		print_status('iteration: ' + str(incv_i) + ', finished step: 6')

		# 8: if i = 1, estimate the noise ratio ε using Eq. (4)
		# evaluate noisy ratio and compute discard ratio
		Num_top=1
		top_pred = np.argsort(y_C2, axis=1)[:,-Num_top:]
		y_true_noisy = np.argmax(asTensor(to_categorical(sequence.pad_sequences(t_C2, maxlen=s_max_length + char_features_C2.shape[1],dtype=float_type),dtype=float_type)),axis=1)
		n_train = len(sentences)
		top_True = np.array([y_true_noisy[i] in top_pred[i,:] for i in range(len(top_pred))])
		if incv_i == 0:
			eval_ratio = 0.001
			product = np.sum(top_True)/(n_train/2)
			e = 1
			while e > product:
				e = ((1-eval_ratio)**2+(eval_ratio**2))
				eval_ratio += 0.001
				if eval_ratio>=1:
					break
				discard_ratio = min(2, eval_ratio/(1-eval_ratio))
		if (remove_ratio == None):
			r = discard_ratio
		else:
			r = remove_ratio
		# 7: Identify n = r|S1| samples that will be removed: R1 = {#n arg maxC2 L(y, f(x; ω))}
		n = int(r*len(S1))
		R1 = []
		t_C2 = asTensor(to_categorical(sequence.pad_sequences(t_C2, maxlen=s_max_length + char_features_C2.shape[1],dtype=float_type),dtype=float_type))
		for i in range(len(y_C2)):
			if (s_lengths_C2[i] == 0): continue
			s_cross_entropy_loss = cross_entropy(t_C2[i], y_C2[i])
			# print(f"{i} - {s_cross_entropy_loss}")
			R1.append((ids_C2[i],None,None,None,s_cross_entropy_loss))
		R1.sort(key=lambda x: x[4], reverse=True)
		R1 = R1[:n]

		print_status('iteration: ' + str(incv_i) + ', finished step: 7')

		# 9: Reinitialize the network f(x; ω)
		model.set_weights(initial_weights) # reset weights
		print_status('iteration: ' + str(incv_i) + ', finished step: 9')


		# 10: Train f(x; ω) on S ∪ C2 for E epochs
		joined_set = join(S,C2)
		print_status('iteration: ' + str(incv_i) + ', finished step: 10 (joining)')


		ids_C2, char_features_C2, word_features_C2, t_C2, _ = extract_features(joined_set)
		char_features_C2 = asTensor(sequence.pad_sequences(char_features_C2, maxlen=s_max_length, dtype=float_type))
		word_features_C2 = asTensor(sequence.pad_sequences(word_features_C2, maxlen=s_max_length, dtype=float_type))
		t_C2 = asTensor(to_categorical(sequence.pad_sequences(t_C2, maxlen=s_max_length + char_features_C2.shape[1],dtype=float_type),dtype=float_type))

		fit_model(model, char_features_C2, word_features_C2, t_C2, main_training=False)
		print_status('iteration: ' + str(incv_i) + ', finished step: 10 (training)')


		# 11: Select samples, S2 = {(x, y) ∈ C1 : y^f = y}
		S2 = []
		ids_C1, char_features_C1, word_features_C1, t_C1, s_lengths_C1 = extract_features(C1)
		char_features_C1 = asTensor(sequence.pad_sequences(char_features_C1, maxlen=s_max_length, dtype=float_type))
		word_features_C1 = asTensor(sequence.pad_sequences(word_features_C1, maxlen=s_max_length, dtype=float_type))

		y_C1 = model.predict([char_features_C1, word_features_C1])
		for i in range(len(y_C1)):
			if (s_lengths_C1[i] == 0): continue
			s = y_C1[i][-s_lengths_C1[i]:] # [[0.2, 0.6, 0.2]... 83 times]
			same = True
			for j in range(len(s)):
				token_prob = s[j]
				predicted_label = 1 if token_prob[1] > token_prob[2] else 2
				label = 1 if t_C1[i][j] == 1 else 2
				if (predicted_label != label):
					same = False
					break
			if (same):
				S2.append((ids_C1[i],char_features_C1[i],word_features_C1[i],t_C1[i],s_lengths_C1[i]))
		print_status('iteration: ' + str(incv_i) + ', finished step: 11')
		

		# 12: Identify n = r|S2| samples that will be removed: R2 = {#n arg maxC1 L(y, f(x; ω))}
		n = int(r*len(S2))
		R2 = []

		t_C1 = asTensor(to_categorical(sequence.pad_sequences(t_C1, maxlen=s_max_length + char_features_C1.shape[1],dtype=float_type),dtype=float_type))
		for i in range(len(y_C1)):
			if (s_lengths_C1[i] == 0): continue
			s_cross_entropy_loss = tf.keras.backend.mean(tf.keras.losses.categorical_crossentropy(t_C1[i], y_C1[i]))
			R2.append((ids_C1[i],None,None,None,s_cross_entropy_loss))

		R2.sort(key=lambda x: x[4], reverse=True)
		R2 = R2[:n]
		print_status('iteration: ' + str(i) + ', finished step: 12')


		# 13: S = S ∪ S1 ∪ S2, C = C − S1 ∪ S2 ∪ R1 ∪ R2
		S = join(S, S1)
		S = join(S, S2)

		to_remove_S = join(S1, S2)
		to_remove_R = join(R1, R2)
		to_remove = join(to_remove_S, to_remove_R)
		C = subtract(C, to_remove)
		print_status('iteration: ' + str(incv_i) + ', finished step: 13')

	# Return the selected set S, remaining candidate set C and estimated noise ratio
	return S, C, eval_ratio

# Load train data
print_status('Loading training data...')
zippath = 'training-data/training-data.zip'
filepath = 'training-data/network_train_' + embedding_type + '_' + lang1 + '_' + lang2 +'.json'
with read_file(filepath,zippath) as train_data_f:
	encoded_sentences = json.load(train_data_f)

x_sentences = []
t_sentences = []
s_max_length = -1

# Read training data
for s in encoded_sentences:
	s = Sentence(encoded_sentence=s)
	s_vectors_only = []
	t_s = []
	char_s = []
	for w in s.words:
		if (ord(w.word[0]) == 65039): continue
		if (w.label != 'other'):
			char_s.append(get_tokenizer_char_features(w.word))
			s_vectors_only.append(w.vector)
			if (w.label == 'lang1'): t_s.append(1)
			if (w.label == 'lang2'): t_s.append(2)
	if (len(t_s) > s_max_length): s_max_length = len(t_s)
	x_sentences.append((s.id, char_s, s_vectors_only, t_s, len(t_s)))
	t_sentences.append(t_s)

nr_batches = int(np.floor(len(x_sentences) / batch_size))
x_sentences = x_sentences[:nr_batches * batch_size]
t_sentences = t_sentences[:nr_batches * batch_size]

model_filepath_1 = 'models/train_models/model_incv_' + lang1 + '_' + lang2 + '_' + architecture + '_' + str(units) + '_' + \
				embedding_type + '_' + optimizer + '_lr' + str(learning_rate) + '_ep' + str(epochs) + '_bsize' + str(batch_size) + \
				'_mom' + str(momentum) + '_incvep' + str(incv_epochs) + '_incviter' + str(incv_iterations) + '_removeratio' + str(remove_ratio) + '_model1.h5'
model_filepath_2 = 'models/train_models/model_incv_' + lang1 + '_' + lang2 + '_' + architecture + '_' + str(units) + '_' + \
				embedding_type + '_' + optimizer + '_lr' + str(learning_rate) + '_ep' + str(epochs) + '_bsize' + str(batch_size) + \
				'_mom' + str(momentum) + '_incvep' + str(incv_epochs) + '_incviter' + str(incv_iterations) + '_removeratio' + str(remove_ratio) + '_model2.h5'
incv_sets_filepath = 'models/train_models/incv_sets_' + lang1 + '_' + lang2 + '_' + architecture + '_' + str(units) + '_' + \
				embedding_type + '_' + optimizer + '_lr' + str(learning_rate) + '_ep' + str(epochs) + '_bsize' + str(batch_size) + \
				'_mom' + str(momentum) + '_incvep' + str(incv_epochs) + '_incviter' + str(incv_iterations) + '_removeratio' + str(remove_ratio) + '.npy'
if(not os.path.exists(model_filepath_1) or not os.path.exists(model_filepath_2)):
	if(not os.path.exists(incv_sets_filepath)):
		print_status("Finding selected and candidate set")
		# Get selected set
		S, C, eval_ratio = INCV(x_sentences, s_max_length)

		# Save S and C
		np.save(incv_sets_filepath,[S,C,eval_ratio],allow_pickle=True)
	else:
		# Load saved sets
		print_status("Loading saved selected and candidate set")
		saved_sets = np.load(incv_sets_filepath, allow_pickle=True)
		S, C, eval_ratio = saved_sets[0], saved_sets[1], saved_sets[2]

	# Compute noise ratio for selected dataset
	class_constant = 1
	noise_ratio_selected = (eval_ratio*eval_ratio) / (eval_ratio*eval_ratio + class_constant*(1-eval_ratio)*(1-eval_ratio))

	# Initialize network 1 (f1) # Note: extract features just for model creation (need the shapes)
	ids_S, char_features_S, word_features_S, t_S, s_lengths_S = extract_features(S)
	char_features_S = asTensor(sequence.pad_sequences(char_features_S, maxlen=s_max_length, dtype=float_type))
	word_features_S = asTensor(sequence.pad_sequences(word_features_S, maxlen=s_max_length, dtype=float_type))
	
	input_char_shape = [char_features_S.shape[1], char_features_S.shape[2]]
	input_word_shape = [word_features_S.shape[1], word_features_S.shape[2]]
	if(architecture == 'bilstm'):
		model1 = BiLSTM(input_char_shape,input_word_shape,optimizer=optimizer,embedding_type=embedding_type,main_output_lstm_units=units,learning_rate=learning_rate,momentum=momentum,return_optimizer=False)
	elif(architecture == 'lstm'):
		model1 = LSTM(input_char_shape,input_word_shape,optimizer=optimizer,embedding_type=embedding_type,main_output_lstm_units=units,learning_rate=learning_rate,momentum=momentum,return_optimizer=False)
	elif(architecture == 'simple_rnn'):
		model1 = SimpleRNN(input_char_shape,input_word_shape,optimizer=optimizer,embedding_type=embedding_type,main_output_lstm_units=units,learning_rate=learning_rate,momentum=momentum,return_optimizer=False)

	# Initialize network 2 (f2) # Note: extract features just for model creation (need the shapes)
	ids_C, char_features_C, word_features_C, t_C, s_lengths_C = extract_features(C)
	char_features_C = asTensor(sequence.pad_sequences(char_features_C, maxlen=s_max_length, dtype=float_type))
	word_features_C = asTensor(sequence.pad_sequences(word_features_C, maxlen=s_max_length, dtype=float_type))

	input_char_shape = [char_features_C.shape[1], char_features_C.shape[2]]
	input_word_shape = [word_features_C.shape[1], word_features_C.shape[2]]
	if(architecture == 'bilstm'):
		model2 = BiLSTM(input_char_shape,input_word_shape,optimizer=optimizer,embedding_type=embedding_type,main_output_lstm_units=units,learning_rate=learning_rate,momentum=momentum,return_optimizer=False)
	elif(architecture == 'lstm'):
		model2 = LSTM(input_char_shape,input_word_shape,optimizer=optimizer,embedding_type=embedding_type,main_output_lstm_units=units,learning_rate=learning_rate,momentum=momentum,return_optimizer=False)
	elif(architecture == 'simple_rnn'):
		model2 = SimpleRNN(input_char_shape,input_word_shape,optimizer=optimizer,embedding_type=embedding_type,main_output_lstm_units=units,learning_rate=learning_rate,momentum=momentum,return_optimizer=False)

	# Number of samples
	train_count = char_features_S.shape[0]
	
	# Warm up epoch
	e_0 = 10

	loss1 = []
	loss2 = []

	# Co-teaching strategy
	for epoch in range(epochs):
		print_status('Start of epoch ' + str(epoch))

		# Iterate over the batches of the dataset.
		for start, end in zip(range(0, train_count, batch_size),
						range(batch_size, train_count + 1, batch_size)):
			# print(f"{start} - {end} : {train_count}")
			# Fetch mini-batch from S
			batch_S = S[start:end]

			# Fetch mini-batch from C
			batch_C = C[start:end]

			if (len(batch_C) == 0): 
				print_status(f"Loss model 1: {loss1} - Loss model 2: {loss2}")
				break

			# Warm-up epoch
			if (epoch <= e_0):
				batch = batch_S
			else:
				batch = np.concatenate([batch_S, batch_C])
			
			n_keep = round(len(batch) * (1 - noise_ratio_selected * min(1, epoch/5)))

			if(n_keep == 0):
				n_keep = len(batch)

			# Extract batch features and prepare data
			ids, char_features, word_features, t, s_lengths = extract_features(batch)
			char_features = asTensor(sequence.pad_sequences(char_features, maxlen=s_max_length, dtype=float_type))
			word_features = asTensor(sequence.pad_sequences(word_features, maxlen=s_max_length, dtype=float_type))
			# t = asTensor(to_categorical(sequence.pad_sequences(t, maxlen=s_max_length + char_features.shape[1])))

			# select samples based on model 1
			# B1 = {#n(e) arg minB L(y, f1(x; ω1))}
			y_pred = model1.predict([char_features, word_features])
			B1 = []
			for i in range(len(y_pred)):
				if (s_lengths[i] == 0): continue
				s = y_pred[i][-s_lengths[i]:]
				s_cross_entropy_loss = 0
				for j in range(len(s)):
					token_prob = s[j]
					predicted_label = 1 if token_prob[1] > token_prob[2] else 2
					label = 1 if t[i][j] == 1 else 2
					cross_entropy_loss = - label * np.log(predicted_label+1e-8)
					s_cross_entropy_loss += cross_entropy_loss
				B1.append((ids[i],char_features[i],word_features[i],t[i],s_lengths[i],s_cross_entropy_loss))
			B1.sort(key=lambda x: x[5], reverse=False)
			B1 = B1[:n_keep]

			# select samples based on model 2
			# B2 = {#n(e) arg minB L(y, f2(x; ω2))}
			y_pred = model2.predict([char_features, word_features])
			B2 = []
			for i in range(len(y_pred)):
				if (s_lengths[i] == 0): continue
				s = y_pred[i][-s_lengths[i]:]
				s_cross_entropy_loss = 0
				for j in range(len(s)):
					token_prob = s[j]
					predicted_label = 1 if token_prob[1] > token_prob[2] else 2
					label = 1 if t[i][j] == 1 else 2
					cross_entropy_loss = - label * np.log(predicted_label+1e-8)
					s_cross_entropy_loss += cross_entropy_loss
				B2.append((ids[i],char_features[i],word_features[i],t[i],s_lengths[i],s_cross_entropy_loss))
			B2.sort(key=lambda x: x[5], reverse=False)
			B2 = B2[:n_keep]

			# Update f1 using B2
			ids_B2, char_features_B2, word_features_B2, t_B2, _ = extract_features(B2)
			char_features_B2 = asTensor(char_features_B2)
			word_features_B2 = asTensor(word_features_B2)
			t_B2 = to_categorical(sequence.pad_sequences(t_B2, maxlen=s_max_length + char_features_B2.shape[1]))
			if(t_B2.shape[2]==2):
				t_B2 = np.insert(t_B2, 0, 0, axis=2)
			t_B2 = asTensor(t_B2)
			loss1 = model1.train_on_batch([char_features_B2, word_features_B2], t_B2)

			# Update f2 using B1
			ids_B1, char_features_B1, word_features_B1, t_B1, _ = extract_features(B1)
			char_features_B1 = asTensor(char_features_B1)
			word_features_B1 = asTensor(word_features_B1)
			t_B1 = to_categorical(sequence.pad_sequences(t_B1, maxlen=s_max_length + char_features_B1.shape[1]))
			if(t_B1.shape[2]==2):
				t_B1 = np.insert(t_B1, 0, 0, axis=2)
			t_B1 = asTensor(t_B1)
			loss2 = model2.train_on_batch([char_features_B1, word_features_B1], t_B1)
			
	# Save models
	model1.save(model_filepath_1)
	model2.save(model_filepath_2)
else:
	print_status('Loading models...')
	model1 = keras.models.load_model(model_filepath_1)
	model2 = keras.models.load_model(model_filepath_2)

# Evaluate the model
# Train accuracy
ids, x_char_features, x_word_features, t, _ = extract_features(x_sentences)
x_char_features = asTensor(sequence.pad_sequences(x_char_features, maxlen=s_max_length, dtype=float_type))
x_word_features = asTensor(sequence.pad_sequences(x_word_features, maxlen=s_max_length, dtype=float_type))
t = asTensor(to_categorical(sequence.pad_sequences(t, maxlen=s_max_length + x_char_features.shape[1])))
_, train_acc_1 = model1.evaluate([x_char_features, x_word_features], t, verbose=1)
_, train_acc_2 = model2.evaluate([x_char_features, x_word_features], t, verbose=1)

# Load dev data
filepath = 'datasets/bilingual-annotated/' + lang1 + '-' + lang2 + '/dev.conll'
val_tokens, val_labels, val_sentences = read_lince_labeled_data(filepath)

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
x_char_test = sequence.pad_sequences(x_char_test, maxlen=s_max_length, dtype=float_type)
x_word_test = sequence.pad_sequences(x_word_test, maxlen=s_max_length, dtype=float_type)
t_test_network = sequence.pad_sequences(t_test, maxlen=s_max_length + x_char_test.shape[1])

# One-hot encode T
t_test_network = to_categorical(t_test_network)

# Convert to tensor
x_char_test, x_word_test, t_test_network = asTensor(x_char_test), asTensor(x_word_test), asTensor(t_test_network)

_, val_acc_1 = model1.evaluate([x_char_test, x_word_test], t_test_network, verbose=1)
_, val_acc_2 = model2.evaluate([x_char_test, x_word_test], t_test_network, verbose=1)
print('Embedding type: %s' % embedding_type)
print('Train accuracy (network 1): %.4f' % train_acc_1)
print('Train accuracy (network 2): %.4f' % train_acc_1)
print('Validation accuracy (network 1): %.4f' % val_acc_1)
print('Validation accuracy (network 2): %.4f' % val_acc_2)

# Get predictions on validation data
model_predictions = model1.predict([x_char_test, x_word_test]) if (val_acc_1 > val_acc_2) else model2.predict([x_char_test, x_word_test])
y = []
token_sentences = []
label_sentences = []
for i in range(len(val_sentences)):
# for i in range(0, 10):
	if(len(val_sentences[i]) > 0):
		tokens, labels = extract_tokens(val_sentences[i])
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
failed_instances_to_csv("incv-instances", token_sentences, y, label_sentences)

t = val_labels
y = [item for y_sent in y for item in y_sent] # Flatten list
val_acc_overall = accuracy_score(t, y)
f1_score_weighted = f1_score(t, y, average='weighted')
f1_score_class = f1_score(t, y, average=None)
print('Validation accuracy (overall): %.4f' % val_acc_overall)
print('Validation F1 score weighted (overall): %.4f' % f1_score_weighted)
print('Validation F1 score per class (overall): %s' % f1_score_class)
