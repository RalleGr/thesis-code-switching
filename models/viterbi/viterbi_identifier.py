import math
import numpy as np

# Source: https://github.com/eginhard/word-level-language-id

from models.viterbi.viterbi_language_model import ViterbiLanguageModel

class ViterbiIdentifier:
	"""Word level language identification.
	"""

	LANG1 = "lang1"
	LANG2 = "lang2"

	# This string should be used for tokens that don't need a language
	# assignment, e.g. punctuation.
	IGNORE = "##IGNORE##"

	def __init__(self,
				 model_file_lang1, model_file_lang2,
				 lex_file_lang1, lex_file_lang2,
				 lex_weight=1):
		"""Initialises the language model.

		Args:
			fr/en: Foreign/English language code.
			model_file_fr/en (str): Foreign/English ViterbiLanguageModel file name.
			lex_file_fr/en (str): Foreign/English Lexicon (1 word + frequency per line).
			lex_weight (float): Weight of the lexicon vs. the character model.
		"""
		self.lex_weight = lex_weight
		self.model = {}
		self.model[self.LANG1] = ViterbiLanguageModel.load(model_file_lang1, lex_file_lang1, lex_weight)
		self.model[self.LANG2] = ViterbiLanguageModel.load(model_file_lang2, lex_file_lang2, lex_weight)

	def identify(self, tokens,
				 transition_probability=0.85,
				 start_probability=0.6):
		"""Context-dependent word level language identification using Viterbi.

		Assigns the most likely language to each token according to both language
		models and the likelihood of switching the language or not.
		"""

		languages = []

		V = [{}] # Stores max probability for each token and language
		S = [{}] # Stores argmax (most likely language)

		# Probability of keeping vs. switching the language
		trans_p = {}
		trans_p[self.LANG1] = {}
		trans_p[self.LANG2] = {}
		trans_p[self.LANG1][self.LANG1] = transition_probability
		trans_p[self.LANG2][self.LANG2] = transition_probability
		trans_p[self.LANG1][self.LANG2] = 1 - transition_probability
		trans_p[self.LANG2][self.LANG1] = 1 - transition_probability

		# Initial probabilities for both languages
		scores = self.score(tokens[0])
		V[0][self.LANG1] = math.log(start_probability) + scores[self.LANG1]
		V[0][self.LANG2] = math.log(1 - start_probability) + scores[self.LANG2]

		langs = [self.LANG1, self.LANG2]
		# Iterate over tokens (starting at second token)
		for t in range(1, len(tokens)):
			V.append({})
			S.append({})
			# Iterate over the two languages
			scores = self.score(tokens[t])
			for lang in langs:
				# Get max and argmax for current position
				term = (V[t-1][lang2] + math.log(trans_p[lang2][lang]) + scores[lang]
						for lang2 in langs)
				maxlang_index, prob = self.max_argmax(term)
				V[t][lang] = prob
				S[t][lang] = langs[maxlang_index] # save lang1 or lang2
			# 	print(prob)
			# print('-----')

		# Get argmax for final token
		languages = [0] * len(tokens)
		languages[-1] = langs[self.max_argmax(V[-1][lang] for lang in langs)[0]] # Use V instead of S to get lang of last token - for 1 word sequences the loop at line 67 is not entered, only have start probabilities, so S is never instantiated with a value

		# Reconstruct optimal path
		for t in range(len(tokens) - 1, 0, -1):
			languages[t-1] = S[t][languages[t]]

		# Sequence confidence score
		confidence_score = self.max_argmax(V[-1][lang] for lang in langs)[1]/len(tokens)

		# Word confidence score
		tokens_confidence_score = []
		for i in range(len(tokens)):
			scores = self.score(tokens[i])
			max_prob = -999
			for lang in langs:
				term = (math.log(trans_p[lang2][lang]) + scores[lang]
						for lang2 in langs)
				maxlang_index, prob = self.max_argmax(term)
				if (prob > max_prob):
					max_prob = prob
			tokens_confidence_score.append(max_prob)

		return languages, confidence_score, tokens_confidence_score

	def max_argmax(self, iterable):
		"""Returns the tuple (argmax, max) for a list,
			where argmax is the index of the maximum value and max is the value itself
		"""
		return max(enumerate(iterable), key=lambda x: x[1])

	def score(self, word):
		"""Returns the weighted log probability according to lexicon + character model.
		"""
		# Punctuation etc. have no influence on the language assignment
		if word == self.IGNORE:
			return {self.LANG1: 1, self.LANG2: 1}

		# Get frequency probability (lex_score) and bigram probability (char_score) for word
		lex_score, char_score = {}, {}
		for lang in [self.LANG1, self.LANG2]:
			lex_score[lang] = math.exp(self.model[lang].lex_score(word))
			char_score[lang] = math.exp(self.model[lang].char_score(word))

		# Relative scores:
		# Lex and char score for a word, divided by the sum of the lex / char scores for EN and ES
		# How likely it is that the word is in EN compared to ES
		lex_score_rel, char_score_rel = {}, {}
		for lang in [self.LANG1, self.LANG2]:
			lex_score_rel[lang] = lex_score[lang] / (lex_score[self.LANG1] +
													 lex_score[self.LANG2])
			char_score_rel[lang] = char_score[lang] / (char_score[self.LANG1] +
													   char_score[self.LANG2])

		weighted_score = {}
		# If neither word is in the lexicon, use only the character model
		if (lex_score[self.LANG1] == math.exp(self.model[self.LANG1].lex_score(ViterbiLanguageModel.OOV)) and
			lex_score[self.LANG2] == math.exp(self.model[self.LANG2].lex_score(ViterbiLanguageModel.OOV))):
			for lang in [self.LANG1, self.LANG2]:
				weighted_score[lang] = math.log(char_score_rel[lang])
		# Else combine both models
		else:
			for lang in [self.LANG1, self.LANG2]:
				weighted_score[lang] = math.log(self.lex_weight * lex_score_rel[lang] +
												(1 - self.lex_weight) * char_score_rel[lang])
		#print word
		#print("%.15f %.15f" % (lex_score[self.LANG1], lex_score[self.LANG2]))
		#print("%.15f %.15f" % (char_score[self.LANG1], char_score[self.LANG2]))
		#print("%.15f %.15f" % (lex_score_rel[self.LANG1], lex_score_rel[self.LANG2]))
		#print("%.15f %.15f" % (char_score_rel[self.LANG1], char_score_rel[self.LANG2]))
		#print("%.15f %.15f" % (weighted_score[self.LANG1], weighted_score[self.LANG2]))
		return weighted_score
