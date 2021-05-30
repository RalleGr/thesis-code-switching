#############################################################
# Source code for paper : A two-step training approach for semi-supervised language identification in code-switched data
# Authors: Dana-Maria Iliescu, Rasmus Grand & Sara Qirko
# IT-University of Copenhagen
# May 2021
#############################################################

from typing import Dict
from typing_extensions import runtime
from tools.arguments import generate_runcommand
from tools.slurm import makefile
import os
import json

EXPERIMENT_MODE = False

## EXPERIMENT CONSTANTS
CONDA_ENV = "saqi"
JOB_NAME = "top-n-language-en-hi-ro"

if not EXPERIMENT_MODE:
	command = generate_runcommand()
	os.system(command)
else:
	with open('config/experiments.json') as f:
		config = json.load(f)
		extra_params = []
		for k in config.keys():
			if len(config[k]) > 1:
				extra_params.append(k)
		command = ""
		if len(extra_params) == 0:
			temp_dict = dict(zip(config.keys(),range(len(config.keys()))))
			for k,v in config.items():
				temp_dict[k] = v[0] if len(v) > 0 else ''
			command = generate_runcommand(temp_dict)
			print(command)
			makefile(JOB_NAME,0,command,CONDA_ENV)
		else:
			for extra_param in extra_params:
				temp_dict = dict(zip(config.keys(),range(len(config.keys()))))
				for k,v in config.items():
					if not k == extra_param:
						temp_dict[k] = v[0] if len(v) > 0 else ''
				for i, v in enumerate(config[extra_param]):
					temp_dict[extra_param] = v
					command = generate_runcommand(temp_dict)
					print(command)
					makefile(f"{JOB_NAME}-{extra_param}",i,command,CONDA_ENV)
