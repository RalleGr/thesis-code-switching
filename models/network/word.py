class Word:
	def __init__(self, word=None, label=None, cs=None, encoded_word=None):
		if encoded_word is None:
			self.word = word
			self.label = label
			self.cs = cs
		else:
			self.json_decode(encoded_word)

	def set_vector(self, vector):
		self.vector = vector.tolist()

	def set_id(self, id):
		self.id = id

	def json_encode(self):
		return self.__dict__
	
	def json_decode(self,obj):
		self.id = obj["id"]
		self.word = obj["word"]
		self.label = obj["label"]
		self.cs = obj["cs"]
		self.vector = obj.get("vector", None)