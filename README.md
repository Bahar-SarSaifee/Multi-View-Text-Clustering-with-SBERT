# Multi-View-Text-Clustering-with-SBERT

I implemented a method for Multi-View text clustering that exploits different text representations to improve the clustering quality.

- **First step:**
  Text representation with pre-trained encoders from **Sentence-BERT**
- **Second step:**
  Clustering Views with the k-means clustering algorithm
- **Third step:**
  Aggregation Partitions with selective voting technique

## Dataset

I use a public dataset from **BBC News datasets** comprised of 2225 articles, each labeled under one of 5 categories: business, entertainment, politics, sport, or tech.

LINK: https://www.kaggle.com/c/learn-ai-bbc/overview

## Source

Hammami, Eya, and Rim Faiz. "Text clustering based on multi-view representations." Proceedings of the 2nd Joint Conference of the Information Retrieval Communities in Europe (CIRCLE 2022), Samatan, Gers, France, July. 2022.
