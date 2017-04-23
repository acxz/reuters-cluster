# MATH 3670 Article Clustering
# Authors: Akash Patel, Nabeel Mahmood, Taylor Fields

import statistics as stat
from nltk.corpus import reuters
import os
import numpy as np
np.set_printoptions(threshold=np.inf)

from scipy.cluster.vq import kmeans, vq

### Get complete word list

os.listdir('/home/apatel435/nltk_data/corpora/reuters/training/')

wordlist = [];
num_articles = 0
artnum = 1000
j = 0
for filename in os.listdir('/home/apatel435/nltk_data/corpora/reuters/training/'):
    if j < artnum:
        words = reuters.words('training/' + str(filename))
        wordlist.append(words)
        num_articles += 1
    j += 1

# Flatten
wordlist = [item for sublist in wordlist for item in sublist]
# Lowercase
wordlist = [x.lower() for x in wordlist]
# Filter repeats
wordlist = list(set(wordlist))
# Sort
wordlist.sort()

### Filter common words and special characters
# take out numbers
wordlist = [ x for x in wordlist if x.isalpha() ]

# Filter common words
common = ["a", "about", "above", "above", "across", "after", "afterwards",
"again", "against", "all", "almost", "alone", "along", "already",
"also","although","always","am","among", "amongst", "amongst", "amount",  "an",
"and", "another", "any","anyhow","anyone","anything","anyway", "anywhere",
"are", "around", "as",  "at", "back","be","became",
"because","become","becomes", "becoming", "been", "before", "beforehand",
"behind", "being", "below", "beside", "besides", "between", "beyond", "bill",
"both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con",
"could", "couldnt", "do", "done", "down",
"due", "during", "each", "e.g.", "eight", "either", "eleven","else",
"elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
"everything", "everywhere", "except", "few", "fifteen", "fifty", "fill", "find",
"fire", "first", "five", "for", "former", "formerly", "forty", "found", "four",
"from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt",
"have", "he", "hence", "her", "here", "hereafter", "hereby", "herein",
"hereupon", "hers", "herself", "him", "himself", "his", "how", "however",
"hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it",
"its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd",
"made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more",
"moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name",
"namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody",
"none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off",
"often", "on", "once", "one", "only", "onto", "or", "other", "others",
"otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per",
"perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed",
"seeming", "seems", "serious", "several", "she", "should", "show", "side",
"since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
"something", "sometime", "sometimes", "somewhere", "still", "such",
"take", "ten", "than", "that", "the", "their", "them", "themselves", "then",
"thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon",
"these", "they", "thick", "thin", "third", "this", "those", "though", "three",
"through", "throughout", "thru", "thus", "to", "together", "too", "top",
"toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up",
"upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever",
"when", "whence", "whenever", "where", "whereafter", "whereas", "whereby",
"wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
"who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within",
"without", "would", "yet", "you", "your", "yours", "yourself", "yourselves",
"the"]

wordlist = [x for x in wordlist if x not in common]


num_wordlist = len(wordlist)

### Continue with matrix building
# Add a 1 if the article contains the word in wordlist
arr = np.zeros((num_wordlist,num_articles))

i = 0

for filename in os.listdir('/home/apatel435/nltk_data/corpora/reuters/training/'):

    if i < artnum:
        words = reuters.words('training/' + str(filename))
        # Lowercase
        words = [x.lower() for x in words]
        # Filter repeats
        words = list(set(words))

        # Add the one if word is in the wordlist
        for x in range(0, num_wordlist):
            if wordlist[x] in words:
                arr[x,i] = 1

    i += 1

stackarr = arr[:,0]

for x in range(1,artnum):
    stackarr = np.vstack((stackarr,arr[:,x]))

# Covariance matrix
cov_arr = np.cov(stackarr)

### Perform k - means algo clustering
# Cluster size chosen to be 5 due to relatively large dataset (1000x1000)
clusters = 5

centroids,_= kmeans(cov_arr,clusters)
idx,_ = vq(cov_arr,centroids)

# Sort matrix by clusters
# ie vectors that are in cluster one are adjacent to one another and so on
# for each cluster

index = idx.argsort();
cov_arrsort = cov_arr[index];

# Transpose matrix to sort the rows
# Matrix is symmetric so transposing it produces the same result
cov_arrtrans = cov_arrsort.transpose();
cluster_cov_arr = cov_arrtrans[index];
print(cluster_cov_arr)
