
# coding: utf-8

# # Sentence Generation Project

# In[ ]:

import numpy as np
from nltk.corpus import BracketParseCorpusReader
from nltk import *
import re


# Constants
PER_SEED = 30
NUM_SEED = 10


# Define the functions to be used

def get_next(distributions, token, order):
    dist = distributions[order-1]
    while ( len(list(dist[token].samples())) == 0):
        order -= 1
        if(order<0 or order>2):
            print("\nERROR: ",token,"HAS NO DISTRIBUTION!!!\n")
        dist = distributions[order-1]
        if (order==1):
            token = token[1]
        elif (order==2):
            token = (token[1], token[2])
            
    return dist[token].generate()


def make_ngram_sentence(distributions, seed, order=1):
    dist_b = distributions[0]
    dist_t = distributions[1]
    dist_f = distributions[2]
    i=0
    sentence = []
    sentence.append(seed)

    root = sentence[i]
    word1 = get_next(distributions, root, 1)
    word2 = get_next(distributions, root, 1)
    i += 1

    if ( dist_b[root].prob(word1) > dist_b[root].prob(word2) ):
        sentence.append(word1)
    else:
        sentence.append(word2)

    if(order == 1):
        while(sentence[i] != "" or len(sentence) <= 3 ):
            root = sentence[i]
            word1 = get_next(distributions, root, order)
            word2 = get_next(distributions, root, order)
            i += 1
        
            if ( dist_b[root].prob(word1) > dist_b[root].prob(word2) ):
                sentence.append(word1)
            else:
                sentence.append(word2)
            
        return sentence
    
    root = (sentence[i-1], sentence[i])
    word1 = get_next(distributions, root, 2)
    word2 = get_next(distributions, root, 2)
    i += 1

    if ( dist_t[root].prob(word1) > dist_t[root].prob(word2) ):
        sentence.append(word1)
    else:
        sentence.append(word2)

    if(order == 2):
        while(sentence[i] != "" or len(sentence) <= 4):
            root = (sentence[i-1], sentence[i])
            word1 = get_next(distributions, root, order)
            word2 = get_next(distributions, root, order)
            i += 1

            if ( dist_t[root].prob(word1) > dist_t[root].prob(word2) ):
                sentence.append(word1)
            else:
                sentence.append(word2)
            
        return sentence

    root = (sentence[i-2], sentence[i-1], sentence[i])
    word1 = get_next(distributions, root, 3)
    word2 = get_next(distributions, root, 3)
    i += 1

    if ( dist_f[root].prob(word1) > dist_f[root].prob(word2) ):
        sentence.append(word1)
    else:
        sentence.append(word2)

    if(order == 3):
        while(sentence[i] != "" or len(sentence) <= 5):
            root = (sentence[i-2], sentence[i-1], sentence[i])
            word1 = get_next(distributions, root, order)
            word2 = get_next(distributions, root, order)
            i += 1

            if ( dist_f[root].prob(word1) > dist_f[root].prob(word2) ):
                sentence.append(word1)
            else:
                sentence.append(word2)
            
        return sentence


def get_ngram_probability(distributions, ngram_input, sentence, order=1):
    dist_b = distributions[0]
    dist_t = distributions[1]
    dist_f = distributions[2]

    prob = np.log(ngram_input.count(sentence[0])/len(ngram_input))
    prob += np.log( dist_b[sentence[0]].prob(sentence[1]) )

    if(order == 1):
        for i in range(2,len(sentence)):
            condition = sentence[i-1]
            prob += np.log(dist_b[condition].prob(sentence[i]))
            
        return prob/len(sentence)
    
    condition = (sentence[0], sentence[1])
    prob += np.log( dist_t[condition].prob(sentence[2]) )

    if(order == 2):
        for i in range(3,len(sentence)):
            condition = (sentence[i-2], sentence[i-1])
            prob += np.log(dist_t[condition].prob(sentence[i]))
            
        return prob/len(sentence)

    condition = (sentence[0], sentence[1], sentence[2])
    prob += np.log( dist_f[condition].prob(sentence[3]) )

    if(order == 3):
        for i in range(4,len(sentence)):
            condition = (sentence[i-3], sentence[i-2], sentence[i-1])
            prob += np.log(dist_f[condition].prob(sentence[i]))
            
        return prob/len(sentence)

def get_next_tag(pos_dist, tag):
    return pos_dist[tag].generate()

def get_next_word(t2w_dist, tag):
    return t2w_dist[tag].generate()

def get_pos_probability(pos_dist, t2w_dist, pos_input, sentence, tags):
    
    prob = np.log(pos_input.count(tags[0])/len(pos_input))
    prob += np.log(t2w_dist[tags[0]].prob(sentence[0]))


    for i in range(1,len(tags)):
        prob += np.log(pos_dist[tags[i-1]].prob(tags[i]))
        prob += np.log(t2w_dist[tags[i]].prob(sentence[i]))

    return prob/len(sentence)


def make_pos_sentence(pos_dist, t2w_dist, w2t_dist, seed):
    i=0
    tag = w2t_dist[seed].generate()

    sentence = []
    sentence.append("")

    tags = []
    tags.append("EOS")

    while(sentence[i] != "" or len(sentence) <= 3 ):
        tags.append(get_next_tag(pos_dist, tags[i]))
        sentence.append(get_next_word(t2w_dist, tags[i+1]))
        i += 1

    return (sentence, tags)


# In[ ]:

# Import and parse the corpus

corpus_root = './corpus_clean/'
corpus = BracketParseCorpusReader(corpus_root, ".*")

tagged_sentences = corpus.tagged_sents()
ngram_input = []
pos_input = []
legal_tags = ["EOS","$","#", "GW", "CC", "CD", "DT", "EX", "FW", "IN", "JJ","JJR","JJS","LS","MD",
             "NN","NNS","NNP",'NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','TO', "UH",'VB',
             'VBD',"VBG","VBN","VBP","VBZ","WDT","WP","WP$","WRB", "\"", "\'", ",", ".", "AFX"]

single_letter_words = ["a", "i", ",", ".", "!", "?", "\'", "\"", ":", ';', '0', '1', '2', "3", '4',
                       '5', "6", '7', '8', "9", "=", "&", "#", '/', '>', "$", '<', '+', '%',]

# tags_removed = ["-NONE-","SYM", "CODE", "ADD", "HYPH","-LSB-", "-RSB-",":", "NFP", "XX", "-LRB-", "-RRB-"]

#  Remove -NONE- and  SYM tags from the training data and create a list of tokens and a list of tags.
for sentence in tagged_sentences:
    for token in sentence:
        word = token[0].lower()
        tag = token[1]
        
        if(tag == "NP"):
            tag = "NNS"

        if not tag in legal_tags:
            del token
            continue
        
        if len(word) == 1:
            if not word in single_letter_words:
                del token
                continue
        
        if (word[0:5] == "rsquo"):
            word = "\'" + word[5:]

        ngram_input.append(word)
        pos_input.append(tag)

    ngram_input.append("")
    pos_input.append("EOS")

unique_alphas = []
unique_tokens = list(set(ngram_input))

for string in unique_tokens:
    if string[0:1].isalpha():
        unique_alphas.append(string)

print("There are",len(ngram_input),"tokens in the corpus.")
print("There are",len(unique_tokens),"unique tokens in the corpus.")
print("There are",len(unique_alphas),"unique tokens that start with a letter.")


tag_set = set(pos_input)
print("There are",len(tag_set),"unique tags in the corpus.")




# In[ ]:

# Create bigram and trigram lists
bgram = list(ngrams(ngram_input,2))
tgram = list(ngrams(ngram_input,3))
fgram = list(ngrams(ngram_input,4))

pos_bgram = list(ngrams(pos_input,2))


# Create conditional frequency distributions
cfd_b = ConditionalFreqDist(bgram)

cfd_t = ConditionalFreqDist()
for trigram in tgram:
    condition = (trigram[0], trigram[1])
    cfd_t[condition][trigram[2]] += 1

cfd_f = ConditionalFreqDist()
for fourgram in fgram:
    condition = (fourgram[0], fourgram[1], fourgram[2])
    cfd_f[condition][fourgram[3]] += 1

cfd_pos = ConditionalFreqDist(pos_bgram)

cfd_t2w = ConditionalFreqDist()
for tag, word in zip(pos_input, ngram_input):
    cfd_t2w[tag][word] += 1

cfd_w2t = ConditionalFreqDist()
for tag, word in zip(pos_input, ngram_input):
    cfd_w2t[word][tag] += 1

# Create conditional probability distributions
cpd_b = ConditionalProbDist(cfd_b, MLEProbDist)
cpd_t = ConditionalProbDist(cfd_t, MLEProbDist)
cpd_f = ConditionalProbDist(cfd_f, MLEProbDist)

cpd_pos = ConditionalProbDist(cfd_pos, MLEProbDist)
cpd_t2w = ConditionalProbDist(cfd_t2w, MLEProbDist)
cpd_w2t = ConditionalProbDist(cfd_w2t, MLEProbDist)


# Consolidate the ngram probability distributions into a single object
distributions = [cpd_b, cpd_t, cpd_f]



print("There are",len(bgram),"bigrams.")
print("There are",len(tgram),"trigrams.")
print("There are",len(fgram),"fourgrams.")



# In[ ]:


seed_list = np.random.choice(unique_alphas,NUM_SEED)

print("Bigrams:\n\n")

for i in range(NUM_SEED):

    sentences = []
    full_sentences = []
    prob = []

    for j in range(PER_SEED):
#        sentences.append(make_ngram_sentence(distributions, seed_list[i], 3))
        sentences.append(make_ngram_sentence(distributions, "", 1))
        prob.append(get_ngram_probability(distributions, ngram_input, sentences[j], 1))
        full_sentences.append(" ".join(sentences[j]))

    best_index = np.argmax(prob)
    print(full_sentences[best_index],"\n",prob[best_index],"\n")


# In[ ]:


seed_list = np.random.choice(unique_alphas,NUM_SEED)

print("Trigrams:\n\n")

for i in range(NUM_SEED):

    sentences = []
    full_sentences = []
    prob = []

    for j in range(PER_SEED):
#        sentences.append(make_ngram_sentence(distributions, seed_list[i], 3))
        sentences.append(make_ngram_sentence(distributions, "", 2))
        prob.append(get_ngram_probability(distributions, ngram_input, sentences[j], 2))
        full_sentences.append(" ".join(sentences[j]))

    best_index = np.argmax(prob)
    print(full_sentences[best_index],"\n",prob[best_index],"\n")


# In[ ]:


seed_list = np.random.choice(unique_alphas,NUM_SEED)

print("Fourgrams:\n\n")

for i in range(NUM_SEED):

    sentences = []
    full_sentences = []
    prob = []

    for j in range(PER_SEED):
#        sentences.append(make_ngram_sentence(distributions, seed_list[i], 3))
        sentences.append(make_ngram_sentence(distributions, "", 3))
        prob.append(get_ngram_probability(distributions, ngram_input, sentences[j], 3))
        full_sentences.append(" ".join(sentences[j]))

    best_index = np.argmax(prob)
    print(full_sentences[best_index],"\n",prob[best_index],"\n")


# In[ ]:

print("POS-Tags:\n\n")

for i in range(len(seed_list)):

    sentences = []
    tags = []
    full_sentences = []
    prob = []

    for j in range(PER_SEED):
        
        (temp_sentence, temp_tags) = make_pos_sentence(cpd_pos, cpd_t2w, cpd_w2t, "")
        sentences.append(temp_sentence)
        tags.append(temp_tags)
    
        prob.append(get_pos_probability(cpd_pos, cpd_t2w, pos_input, sentences[j], tags[j] ))
        pos_sent = " ".join(sentences[j])
#        print(pos_sent,"\n",prob[j],"\n")
    
    
    best_index = np.argmax(prob)
    pos_sent = " ".join(sentences[best_index])
    print(pos_sent,"\n",prob[best_index],"\n")



# In[ ]:



