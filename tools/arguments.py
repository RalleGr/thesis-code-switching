import os
import argparse
import platform

def generate_runcommand(config=None):
	python_command = 'python3'
	if platform.system()  == 'Windows' and config is None:
		python_command = 'python'

	if not config:
		arg_parser = argparse.ArgumentParser()

		# LANGUAGE
		arg_parser.add_argument("lang1",type=str,help="Language code for language one, choose from [en,es,hi,ne,ar,arz]")
		arg_parser.add_argument("lang2",type=str,help="Language code for language two, choose from [en,es,hi,ne,ar,arz]")

		# STANDARD HYPERPARAMETERS
		arg_parser.add_argument("model",type=str,help="Choose one model of [baseline,top_n,incv,idn]")
		arg_parser.add_argument("architecture",type=str,help="Choose one architecture of [bilstm,lstm,simple_rnn]")
		arg_parser.add_argument("units",type=str,help="Select a number of units")
		arg_parser.add_argument("embedding",type=str,help="Choose one of the following embeddings [fasttext_bilingual,fasttext_bilingual_concatenated_articles,fasttext_bilingual_concatenated_tweets,bert]")
		arg_parser.add_argument("optimizer",type=str,help="Choose one of the following optimizers [sgd,adam]")
		arg_parser.add_argument("learning_rate",type=str,help="Choose a learning rate for the optimizer [range 0.0 to 1.0]")
		arg_parser.add_argument("epochs",type=str,help="Select an amount of epochs [20,30,40,50,60,70]")
		arg_parser.add_argument("batch_size",type=str,help="Choose a batch size [8,16,32,64]")

		# OPTIONAL HYPERPARAMETERS
		arg_parser.add_argument("--momentum",type=str,help="Choose a momentum [range 0.0 to 1.0]")
		arg_parser.add_argument("--gpu",dest='gpu',action='store_true',help="Use GPU hardware resources")
		arg_parser.add_argument("--log",dest='log',action='store_true',help="Use Tensorflow logging callback for profiling")
		arg_parser.add_argument("--mixed",dest='mixed',action='store_true',help="Use mixed precision e.g float16 when running on GPU")

		# SPECIFIC HYPERPARAMETERS
		## Top N
		arg_parser.add_argument("--nr_of_batches",type=str,help="Choose a number of batches [between 1 and 524]")

		## INCV
		arg_parser.add_argument("--incv_epochs",type=str,help="Choose a number of INCV epochs [range 1 to 100]")
		arg_parser.add_argument("--incv_iter",type=str,help="Choose a number of INCV iterations [range 1 to 4]")
		arg_parser.add_argument("--remove_ratio",type=str,help="Choose a remove ratio [range 0.0 to 1.0]")

		## IDN
		arg_parser.add_argument("--eta_lr",type=float,help="Choose a learning rate for eta [in paper code: 0.05]")
		arg_parser.add_argument("--pretrained_model_name",type=str,help="Give the name of a pre-trained model for calculating label noise probabilities.")

		# Parse args
		args = arg_parser.parse_args()

		optional_param_string = f"{'--gpu' if args.gpu else ''} {'--log' if args.log else ''} {'--mixed' if args.mixed else ''} {'--momentum ' + args.momentum if args.momentum is not None else ''}"

		run_command = ""

		# Create python run command with args
		if(args.model == 'baseline'):
			run_command = f"{python_command} code_switching_network_baseline.py {args.lang1} {args.lang2} {args.architecture} {args.units} {args.embedding} {args.optimizer} {args.learning_rate} {args.epochs} {args.batch_size} {optional_param_string}"
		elif (args.model == 'top_n'):
			run_command = f"{python_command} code_switching_network_top_n.py {args.lang1} {args.lang2} {args.architecture} {args.units} {args.embedding} {args.optimizer} {args.learning_rate} {args.epochs} {args.batch_size} {optional_param_string} {'--nr_of_batches ' + args.nr_of_batches if args.nr_of_batches is not None else ''}"
		elif (args.model == 'incv'):
			optional_param_string_incv = f"{'--incv_epochs ' + args.incv_epochs if args.incv_epochs is not None else ''} {'--incv_iter ' + args.incv_iter if args.incv_iter is not None else ''} {'--remove_ratio ' + args.remove_ratio if args.remove_ratio is not None else ''}"
			run_command = f"{python_command} code_switching_network_incv.py {args.lang1} {args.lang2} {args.architecture} {args.units} {args.embedding} {args.optimizer} {args.learning_rate} {args.epochs} {args.batch_size} {optional_param_string} {optional_param_string_incv}"
		elif(args.model == 'idn'):
			optional_param_string_idn = f"{'--eta_lr ' + args.eta_lr if args.eta_lr is not None else ''} {'--pretrained_model_name ' + args.pretrained_model_name if args.pretrained_model_name is not None else ''}"
			run_command = f"{python_command} code_switching_network_idn.py {args.lang1} {args.lang2} {args.architecture} {args.units} {args.embedding} {args.optimizer} {args.learning_rate} {args.epochs} {args.batch_size} {optional_param_string} {optional_param_string_idn}"

		return run_command
	else:
		return createConfigCommand(python_command,config)

def createConfigCommand(python_command, config):
	optional_param_string = f"{'--gpu' if config['gpu'] == '1' else ''} {'--log' if config['log'] == '1' else ''} {'--mixed' if config['mixed'] == '1' else ''} {'--momentum ' + config['momentum'] if config['momentum'] != '' else ''}"
	# Create python run command with args
	if(config['model'] == 'baseline'):
		run_command = f"{python_command} code_switching_network_baseline.py {config['lang1']} {config['lang2']} {config['architecture']} {config['units']} {config['embedding']} {config['optimizer']} {config['learning_rate']} {config['epochs']} {config['batch_size']} {optional_param_string}"
	elif (config['model'] == 'top_n'):
		run_command = f"{python_command} code_switching_network_top_n.py {config['lang1']} {config['lang2']} {config['architecture']} {config['units']} {config['embedding']} {config['optimizer']} {config['learning_rate']} {config['epochs']} {config['batch_size']} {optional_param_string} {'--nr_of_batches ' + config['nr_of_batches'] if config['nr_of_batches'] is not None else ''}"
	elif (config['model'] == 'incv'):
		optional_param_string_incv = f"{'--incv_epochs ' + config['incv_epochs'] if config['incv_epochs'] is not None else ''} {'--incv_iter ' + config['incv_iter'] if config['incv_iter'] is not None else ''} {'--remove_ratio ' + config['remove_ratio'] if config['remove_ratio'] is not None else ''}"
		run_command = f"{python_command} code_switching_network_incv.py {config['lang1']} {config['lang2']} {config['architecture']} {config['units']} {config['embedding']} {config['optimizer']} {config['learning_rate']} {config['epochs']} {config['batch_size']} {optional_param_string} {optional_param_string_incv}"
	elif(config['model'] == 'idn'):
		optional_param_string_idn = f"{'--eta_lr ' + config['eta_lr'] if config['eta_lr'] is not None else ''} {'--pretrained_model_name ' + config['pretrained_model_name'] if config['pretrained_model_name'] is not None else ''}"
		run_command = f"{python_command} code_switching_network_idn.py {config['lang1']} {config['lang2']} {config['architecture']} {config['units']} {config['embedding']} {config['optimizer']} {config['learning_rate']} {config['epochs']} {config['batch_size']} {optional_param_string} {optional_param_string_idn}"

	return run_command