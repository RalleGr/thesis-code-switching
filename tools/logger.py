import sys
from datetime import datetime

class Logger:
	def __init__(self,name):
		self.backup = sys.stdout
		self.name = name
	
	# Begin logging to file
	def begin(self):
		now = datetime.now()
		date = now.strftime("%m-%d-%Y")
		self.log = open(f"logs/{date}-{self.name}.log", "a",encoding="utf-8")
		self.terminal = sys.stdout
		sys.stdout = self
	
	# Close log file stream and return to old stdout
	def end(self):
		self.log.write("-"*100+"\n")
		self.log.close()
		sys.stdout = self.backup

	# Duplicate prints to both stdout and log file
	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)

	def flush(self):
		self.terminal.flush()

		