import csv
import os
import regex
import emoji
import string
import json
import fasttext
from datetime import datetime
# from models.network.word import Word
from zipfile import ZipFile
from tools.colors import Colors
import tensorflow as tf
import numpy as np

def subtract(A,B):
	result = []
	for el_A in A:
		filtered = list(filter(lambda x: (x[0] == el_A[0]), B))
		if (len(filtered) == 0):
			result.append(el_A)
	return result

def join_R(A, B):
	A = A.tolist()
	B = B.tolist
	joined = A.copy()
	for el_B in B:
		foundAny = False
		for x in joined:
			if(x[0] == el_B[0]):
				foundAny = True
				break
		if (not foundAny):
			joined.append(el_B)
	return joined

def join(A, B):
	joined = A.copy()
	for el_B in B:
		foundAny = False
		for x in joined:
			if(x[0] == el_B[0]):
				foundAny = True
				break
		if (not foundAny):
			joined.append(el_B)
	return joined

def join_np(A, B):
	joined = A.copy()
	for el_B in B:
		foundAny = False
		for x in joined:
			if(x[0] == el_B[0]):
				foundAny = True
				break
		if (not foundAny):
			np.hstack((joined,el_B))
	return joined


def write_dict(DICTIONARIES_PATH, frequency_dict, freq_dict_filename, probability_dict=None, probability_dict_filename=''):
	frequency_dict_csv = csv.writer(open(DICTIONARIES_PATH + freq_dict_filename + '.csv', 'w', encoding='UTF-16'))
	frequency_dict_csv.writerow(['word', 'frequency'])
	for key, val in frequency_dict.items():
		frequency_dict_csv.writerow([key, val])

	if probability_dict is not None:
		probability_dict_csv = csv.writer(open(DICTIONARIES_PATH + probability_dict_filename + '.csv', 'w', encoding='UTF-16'))
		probability_dict_csv.writerow(['word', 'probability'])
		for key, val in probability_dict.items():
			probability_dict_csv.writerow([key, val])

# Punctuation, Numbers and Emojis
def is_other(word):
	if word[0] == '#':
		word = word.split('#')[1]

	def isfloat(value):
		try:
			float(value)
			return True
		except ValueError:
			return False

	if (len(word)==1 and ord(word) == 65039):
		return True
	
	special_punct = ['¡', '¿', '“', '”', '…', "'", '']
	for sp in special_punct:
		if word == sp:
			return True

	numeric_sufixes = ['st', 'nd', 'rd', 'th']
	if(word[-2:] in numeric_sufixes and isfloat(word[:-2])):
		return False

	if word.isnumeric() or isfloat(word):
		return True

	if '\n' in word or '\t' in word:
		return True

	digits = '0123456789'
	for c in word:
		if (c in string.punctuation or c in digits) and c != "\'":
			return True
	
	flags = regex.findall(u'[\U0001F1E6-\U0001F1FF]', word)
	data = regex.findall(r'\X', word)
	for word in data:
		if any(char in emoji.UNICODE_EMOJI for char in word) or '♡' in word or len(flags) > 0:
			return True

	return False

# Python code to merge dict using update() method
def merge_dictionaries(dict1, dict2):
	dict2.update(dict1)
	return dict2

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def print_status(status):
	now = datetime.now()
	current_time = now.strftime("%H:%M:%S")
	tf.print(f"{Colors.OKGREEN}[{current_time}]{Colors.ENDC} {status}")

def save_predictions(predictions, file_name):
	"""Saves the language model to the specified file in JSON format"""
	if (type(predictions) == list):
		with open(file_name, 'w', encoding='utf-8') as f:
			for sentence in predictions:
				for label in sentence:
					f.write("%s\n" % label)
				f.write("\n")
	else:
		with open(file_name, 'w', encoding='utf-8') as f:
			json.dump(predictions, f)

	print_status('Predictions saved at: ' + file_name)

def save_train_data(data, file_name):
	"""Saves the language model to the specified file in JSON format"""
	if (type(data) == list):
		print_status(file_name)
		with open(file_name, 'w', encoding='utf-8') as f:
			for sentence in data:
				for word in sentence:
					# if (is_other(word[0])): continue
					f.write(f"{word[0]} ")
				f.write("\n")

	print_status('Data saved at: ' + file_name)

# Read data from original test set
def read_lince_unlabeled_data(filepath, vectorize=False, embeddings_filepath=None):
	model = None
	if (vectorize): model = fasttext.load_model(embeddings_filepath)
	
	file = open(filepath, 'rt', encoding='utf8')
	words = []
	s = []
	sentences = []
	for line in file:
		# Remove empty lines, lines starting with # sent_enum, \n and split on tab
		if (line.strip() is not ''):
			token = line.rstrip('\n')
			s.append(token.lower())
		else:
			sentences.append(s)
			s = []
	
	file.close()
	return words, sentences

# Read data from train/dev set
def read_lince_labeled_data(filepath):
	file = open(filepath, 'rt', encoding='utf8')
	sentences = []
	words = []
	labels = []
	for line in file:
		# Remove empty lines, lines starting with # sent_enum, \n and split on tab
		if (line.strip() != ''):
			if ('# sent_enum' in line):
				s = []
			else:
				line = line.rstrip('\n')
				splits = line.split("\t")
				if (splits[1]=='ambiguous'
					or splits[1]=='fw'
					or splits[1]=='mixed'
					or splits[1]=='ne'
					or splits[1]=='unk'
				):
					continue
				else:
					word = splits[0].lower()
					label = splits[1]

					words.append(word)
					labels.append(label)
				
					s.append((word, label))
		else:
			sentences.append(s)

	file.close()
	return words, labels, sentences

def read_file(internal_path,zip_path=None,mode='rt',encoding='utf-8'):
	if(zip_path is not None and os.path.exists(zip_path)):
		print_status("-- reading data from Zip")
		with ZipFile(zip_path, 'r') as zip:
			f = zip.open(internal_path)
			return f
	else:
		print_status("-- reading uncompressed data")
		f = open(internal_path,mode=mode,encoding=encoding)
		return f

def failed_instances_to_csv(filename, sentences, predicted_labels, true_labels):
	with open(f"./predictions/{filename}.csv", "w", encoding="utf-8") as f:
		f.write("SentenceIndex;Word;Predicted Labels;True Labels;Correctness\n")
		for i, sentence in enumerate(sentences):
			has_wrong_instance = False
			for j, word in enumerate(sentence):
				if predicted_labels[i][j] != true_labels[i][j]:
					has_wrong_instance = True
			if has_wrong_instance:
				for j, word in enumerate(sentence):
					f.write(f"{i};{word};{predicted_labels[i][j]};{true_labels[i][j]};{predicted_labels[i][j] == true_labels[i][j]}\n")
	f.close()
