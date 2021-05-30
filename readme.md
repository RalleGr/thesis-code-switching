# A two-step training approach for semi-supervised language identification in code-switched data
The source code for the masters thesis called : A two-step training approach for semi-supervised language identification in code-switched data by authors Dana-Maria Iliescu, Rasmus Grand & Sara Qirko - IT-University of Copenhagen - May 2021.

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
3. Train the embeddings using (choose desired embedding block by commenting it out)
```python
python3 train_embeddings.py [lang1] [lang2]
```
4. Run viterbi to create training data
```python
python3 code_switching_viterbi.py [lang1] [lang2] [train|train_dev] [embedding]
```

## Train a model
Train a model using the following
```python
python3 train.py [lang1] [lang2] [model] [architecture] [units] [embedding] [optimizer] [learning_rate] [epochs] [batch_size] [optional parameters]
```
Possible parameters as well as optional parameters is revealed by 
```python 
python3 train.py -h
```

## Test predictions
```python 
python3 test_networks.py [lang1] [lang2] [embedding] [model name]
```
