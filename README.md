# horror-author-identification
Spooky Author Identification

Team Name: Those Meddling Kids
Team Members: Osman Malik, Kyle Cordes, Hasan Demirci, Akhil Dixit, Santosh Devarakonda
## Objective
Explore the suitability of Naive Bayes, LSTM and CNN classifiers in predicting the author of excerpts from horror stories by Edgar Allan Poe, Mary Shelley, and HP Lovecraft.

Source: https://www.kaggle.com/c/spooky-author-identification

## Areas of Exploration
Feature Extraction: Bag of words, Word Embedding
Learning Algorithms: (Multinomial) Naive Bayes baseline, LSTM, CNN 
Evaluation Criteria

## Methodology
### Feature Extraction
Bag of Words
- TF-IDF
- N-grams

Word Embedding
- Pre-trained word vectors (GloVe, word2vec)
- Train own word embedding?

### Classifiers
For each classifier algorithm, we want to explore the effect of their inherent hyperparameters on the evaluation criteria.

#### Naive Bayes
http://www.statsoft.com/textbook/naive-bayes-classifier
https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html
http://scikit-learn.org/stable/modules/naive_bayes.html
Suggest Multinomial Naive Bayes but can check out other event models such as Poisson

#### LSTM Hyperparameters
https://deeplearning4j.org/lstm.html
- Optimizer Algorithm
- Learning Rate
- Learning Iterations (# of epoch and early stopping)
- Batch size
- Stacking Layers

#### CNN Hyperparameters
http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/
- Convolution width (narrow vs wide)
- Stride size of kernel (filter)
- Pooling Layers
- Channels e.g. instead of RGB for images, we have different word embeddings e.g. word2vec and GloVe
