from models.network.word import Word
class Sentence:
	def __init__(self, words=None, sentence_cs=None, encoded_sentence=None):
		if(encoded_sentence is None):
			self.words = words
			self.sentence_cs = sentence_cs
		else:
			self.json_decode(encoded_sentence)

	def add_word(self, word):
		self.words.append(word)

	def set_id(self, id):
		self.id = id

	def json_encode(self):
		encoded_words = [w.json_encode() for w in self.words]
		return dict(id=self.id, words = encoded_words, sentence_cs = self.sentence_cs)

	def json_decode(self, obj):
		self.id = obj["id"]
		self.words = [Word(encoded_word=w) for w in obj['words']]
		self.sentence_cs = obj['sentence_cs']

	def to_string():
		return [w for w.word in self.words]
		
	