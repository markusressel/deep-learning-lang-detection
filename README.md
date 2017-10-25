# Deep Learning Programming Language Detection
Deep Learning using Keras (and Tensorflow) to detect programming language of a code file (currently achieves 97.1% accuracy). Approach for DL has been based on Zhang and LeCun's 2016 [paper](https://arxiv.org/pdf/1502.01710.pdf) "Text Understanding from Scratch". The main technique is that instead of using word2vec to create word embedding, characters are quantised (turned to one-hot-vector) and then the document is represented by a sequence of quantised characters (vectors). Currently the document is truncated at 2KB and smaller docs are padded by all zero vectors. 

The code from @keon's Github [project](https://github.com/keon/keras-text-classification) was very helpful to get started and I used the some of the same structure.

## Purpose
The project is set up to demonstrate - in simple steps - how to use Keras to build a Deep Learning (DL) model to reconise programming language of a code file. This can be used to detect language of code snippets but needs to be trained for it - the dataset provided with the project can be turned to snippets and then add to the training set.  

## Languages
15 language was chosen as below:
XML, F#, Clojure, Java, Scala, Python, Javascript, C++, CSS, C#, go, PHP, Ruby, Powershell and bash

## Data
All the data was extracted from Github using its search feature. Common inert words used for search where the word is not a keyword in a language. The words used are:

> load, dump, write, stream, api, manage, broker, save, process, service
mapping, dispatch, copy, duplicate, sample, chunk, instrument
calculate, append, repository, facade, handler, message, invoke,
controller, locator, customer, view, model, nav, show, new, old, legacy

Data are separated into training (~2000 per language) and test (~1000).

## Run the code
You need Keras (and TensorFlow). The code is in python 2.7.

### Training

To train, simply run the `train.py` script:

```python
python train.py
```

If you have GPU enabled, it will automatically use it.

### Testing in batch mode

```python
python test_run.py
```

### Testing a single file

```python
python test.py mytestfile.something
```

This will print the result with likelihood for each 

```buildoutcfg
1/1 [==============================] - 0s
[[  1.01206303e-02   2.55753571e-06   1.85673562e-05   8.22590722e-04
    1.82982374e-04   5.92526863e-04   2.45171529e-03   3.59758809e-02
    3.59477235e-05   9.33431264e-04   1.38449148e-04   5.55862606e-01
    3.90967846e-01   5.18163608e-04   1.36034854e-03   1.56738879e-05]]
  - go:     		1.0%
  - csharp:     	0.0%
  - java:     		0.0%
  - js:     		0.0%
  - cpp:     		0.0%
  - ruby:     		0.0%
  - powershell:         0.0%
  - bash:     		4.0%
  - php:     		0.0%
  - css:     		0.0%
  - xml:     		0.0%
* - python:     	56.0%
  - scala:     		39.0%
  - clojure:     	0.0%
  - fsharp:     	0.0%
```

## Results

System currently achieves **97.1% accuracy** on test data (not seen).
