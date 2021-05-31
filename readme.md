# A two-step training approach for semi-supervised language identification in code-switched data
The source code for the masters thesis called : A two-step training approach for semi-supervised language identification in code-switched data by authors Dana-Maria Iliescu (dail@itu.dk), Rasmus Grand (gran@itu.dk) & Sara Qirko (saqi@itu.dk) - IT-University of Copenhagen - May 2021. 

This project was supervised by Rob van der Goot (robv@itu.dk).

## Requirements
Python

- Python 3.7.4

Requirements in file: [requirements.txt](requirements.txt)
- lxml==4.5.2
- tensorflow==2.1.0
- spacy==2.3.2
- fasttext==0.9.2
- h5py==2.10.0
- emoji==0.6.0
- numpy==1.17.3
- transformers==3.1.0
- regex==2021.4.4.
- scikit-learn==0.24.2
- nltk==3.5
- lxml==4.5.2

For using embeddings from pre-trained BERT models, install:
- tensorflow==2.2.0
- git+https://github.com/huggingface/transformers

Requirements for CUDA:
https://www.tensorflow.org/install/gpu#software_requirements

## Train viterbi and create embeddings
1. Choose a language pair e.g ```en``` and ```es```
2. Run the following files for training viterbi using :
```python
python3 train_viterbi.py [lang1]
python3 train_viterbi.py [lang2]
```
This will generate two language models (frequency dictionaries) for Viterbi to use. The generated dictionaries will be saved as ```.csv``` in the ``dictionaries`` directory.

3. Train the embeddings using (choose desired embedding block by commenting it out)
```python
python3 train_embeddings.py [lang1] [lang2]
```
This command will create an embeddings file and save it in the ``embeddings`` directory.

4. Run viterbi to create training data
```python
python3 code_switching_viterbi.py [lang1] [lang2] [train|train_dev] [embedding]
```
Viterbi will output predictions along with confidence scores for the predictions. This will all be contained in a ``.json`` file saved in the ``training-data`` directory.

## Train a model
Train a model using the following
```python
python3 train.py [lang1] [lang2] [model] [architecture] [units] [embedding] [optimizer] [learning_rate] [epochs] [batch_size] [optional parameters]
```

Example:
```
python3 train.py en es baseline bilstm 64 fasttext_tweets_concatenated_shuffled adam 0.03 50 32
```

Possible parameters as well as optional parameters is revealed by 
```python 
python3 train.py -h
```

When the model finishes training, it will be saved in the ``models`` directory, in the subdirectory called ``train_models``

## Test models
To test the models use the following: 
```python 
python3 test_networks.py [lang1] [lang2] [dev|test] [embedding] [model name]
```
Predictions are saved in the ``predictions`` directory where sub directories are (depending on dataset used) ``dev`` or ``test``, and inside one of these subdirectories a folder named after the respective language pair e.g ``en-es``.

- For the ``dev`` dataset, output prediction files have both words and predictions given by the model. 
- For the ``test`` dataset, output prediction files only contain the predictions given by the model.

## Experimentation using Slurm
The ``config`` directory contains configuration files that can be used for creating many jobs for experimentation of parameters. The [experiments.json](config/experiments.json) contains lists of parameters, that can be used for generating many slurm jobs. If one wishes to use this, then the train.py needs to have the ``EXPERIMENT_MODE=True`` as well as the constants for ``CONDA_ENV`` and ``JOB_NAME``. Also the [slurm-config.json](config/slurm-config.json) has to contain slurm specific parameters for the job files. When this has been changed, the behavior of train.py will change to take parameters from the json files instead and only need to be called using :

```
python3 train.py
```

This will generate job files with all parameter combinations in the ``jobs`` directory.


_Happy code-switching_



