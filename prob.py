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
artnum = 10
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

#print(wordlist)
#print('     ')

### TODO Filter common words and special characters
### can based it on words that only occur once
# take out numbers
wordlist = [ x for x in wordlist if x.isalpha() ]

### ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#print(wordlist)

num_wordlist = len(wordlist)

### Continue with matrix building

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
#print("Covariance")
print(cov_arr)
print(" ")

### Perform k - means algo clustering

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
cov_arrsort = cov_arrtrans[index];
print(cov_arrsort)
