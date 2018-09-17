# Summary

This project looks at classifying news articles to various topics. The purpose of the project is to use the document classifications and the historical frequency of article publications on these topics to correlate to price. News articles will be taken from CryptoCompare's news API feature.  

## Requirements

The GuidedLDA module is required to use this code. 
It can be found here: https://github.com/vi3k6i5/GuidedLDA
It can be installed using `pip install guidedlda`

This code also requires pandas and sklearn, which I recommend downloading in the Anaconda distribution. 

# Development

### LDA

The topics were initially be defined using Latent Dirichlet Allocation (LDA). The implementation of LDA from sklearn was used, which uses online variational Bayes optimisation. The experimentation was primarily carried out in the Jupyter notebook. It was found that the topics that were found were hard to differentiate from each other and classify as a particular topic e.g. 'Bitcoin'. As the topic areas generally have a lot of overlap the exact number of topics required was hard to decide upon. The project considered choosing number of topics by minimising per word perplexity but this did not give easily interpretable topics. Reducing the LDA hyperparameters in the form of the Dirichlet priors to give more specific document classifications also did not achieve more meaningful topics.   

### Guided LDA

The idea to improve interpretability of topics was to instead 'seed' the classification with an asymmetric Dirichlet prior biased towards words which are associated with particular topics. A useful package written by [2] was used to implement this. The seed words and topics were drawn from the current primitive news classification used on the API. These topics and seed words were then optimised by manually adjustments based on topic classification results. 

### Using the model

The topic allocations can be imported into a pandas dataframe and analysed by looking at prevalence of allocation to a topic over time. However, we could be a bit smarter and pickup the reserach direction that is taken in [1] with their sparse PCA model. The idea is that we want to be able to query a search using a list of words that define the topic of interest. This can be achieved by generating a word list from word->document assignment probabilities for each document, and then calculating cosine similarity of the search terms to each document at every time point. 

# Future work 
 
There is scope for further optimisation for a number of hyperparameters:

1. Changing base symmetrical Dirichlet prior variables alpha and eta
2. Changing the seed confidence
3. Merging topics which share a large number of the same highest probability words 

Implement cosine similarity metric.

# References 
 
[1] Strong correlations were found for topic publication frequency and voting preference during the US presidential election. 
https://www.sciencedirect.com/science/article/pii/S2405896317331993/pdf?md5=e03cd91a8bdb54d95825df10ffcf6a1f&pid=1-s2.0-S2405896317331993-main.pdf
This method uses sparse PCA which they claim gives more interpretable groupings than LDA.

[2] Implementing "Guided" LDA for more interpretable document topics:
https://medium.freecodecamp.org/how-we-changed-unsupervised-lda-to-semi-supervised-guidedlda-e36a95f3a164
