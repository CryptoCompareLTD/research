# Reddit sentiment analysis

## Summary 

This project uses the NLTK Python package to classify article headlines from the Reddit API. The VADER sentiment analysis is used to analyse sentiment of the article headers in the form of a compound score in the range [-1, 1]. A separate logistic regression is used with manually labelled headlines and the sentiment score to train the boundary of classification for negative and positive articles. Main exploration is in the Jupyter Notebook - for reasons listed below the project was not fully implemented in Python scripts.  

## Findings

VADER-classified news article headlines do not appear to correlate strongly with manually labelled headlines. This is due to the difficulty of separating content and sentiment. The vast majority of headlines are written in a neutral tone as informative news pieces, which leads to the neutral class being by far the most popular. Also,an article can be clearly a 'negative' one, but the phrasing can lead VADER to classify it as positive e.g. "Bitcoin crash is good news for shorts". For using news data to predict price we would clearly want this headline to throw a negative flag, but without understanding of content we simply recieve that the sentiment was positive.  

## Future work 

Sentiment doesn't seem to be exactly what we are after - maybe looking at topic is more interesting. Discovering what the main topic classes are could allow for improved tagging on the API, and the topics could then be linked to price movements. In fact, if we dont want to (or don't know how to) classify topics as good or bad, we could look at what the movement of price was like during increased frequency of articles about that topic. This suggests that the next step is a clustering algorithm.  

## References 

Adapted from:
https://www.learndatasci.com/tutorials/predicting-reddit-news-sentiment-naive-bayes-text-classifiers/

NLTK:
Bird, Steven, Edward Loper and Ewan Klein (2009),
Natural Language Processing with Python. O'Reilly Media Inc.

VADER:
Hutto, C.J. & Gilbert, E.E. (2014). 
VADER: A Parsimonious Rule-based Model for Sentiment Analysis
of Social Media Text. Eighth International Conference on
Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
