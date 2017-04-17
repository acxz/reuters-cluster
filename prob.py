import statistics as stat
from nltk.corpus import reuters
import os
import numpy as np
np.set_printoptions(threshold=np.inf)

from scipy.cluster.vq import kmeans, vq


### Swap rows and columns
def swap_rows(C, var1, var2):
    '''
    Function to swap two rows in a covariance matrix,
    updating the appropriate columns as well.
    '''
    D = C.copy()
    D[var2, :] = C[var1, :]
    D[var1, :] = C[var2, :]

    E = D.copy()
    E[:, var2] = D[:, var1]
    E[:, var1] = D[:, var2]

    return E




### Get complete word list

os.listdir('/home/apatel435/nltk_data/corpora/reuters/training/')

wordlist = [];
num_articles = 0
artnum = 3
j = 0
for filename in os.listdir('/home/apatel435/nltk_data/corpora/reuters/training/'):
    if j < artnum:
        words = reuters.words('training/' + str(filename))
        #print(str(i) + '-' + str(words));
        wordlist.append(words)
        num_articles += 1
    j += 1

# Flatten
wordlist = [item for sublist in wordlist for item in sublist]
# Lowercase
wordlist = [x.lower() for x in wordlist]
# Filter repeats
wordlist = list(set(wordlist))

### TODO Filter common words and special characters
### can based it on words that only occur once


### ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Sort
wordlist.sort()

num_wordlist = len(wordlist)
#print(wordlist)

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

        # Add the one
        for x in range(0, num_wordlist):
            if wordlist[x] in words:
                arr[x,i] = 1

    i += 1

stackarr = arr[:,0]

for x in range(1,artnum):
    stackarr = np.vstack((stackarr,arr[:,x]))

cov_arr = np.cov(stackarr)
print(cov_arr)

### Perform k - means algo clustering

kmeaned = kmeans(cov_arr,1)
centroids = kmeaned[0]

index = centroids.argsort()

print(index)




### Vector quantization

#vecquan = vq(cov_arr,centroids)
#print(vecquan)

