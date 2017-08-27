import os
import json
from pprint import pprint
from graphviz import Digraph
from hmldb import HmlDB
import random
from itertools import islice
import pickle


def read(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def write(filename, trie):
    with open(filename, 'wb') as f:
        pickle.dump(trie, f, pickle.HIGHEST_PROTOCOL)


def put(trie, words):
    id = 0
    for word in words:
        current = trie[0]
        numberOfEqualLetters = 0
        for j in range(len(word)-1, -1, -1):
            if word[j] in current:
                numberOfEqualLetters += 1
                if((numberOfEqualLetters == len(word)) and current[word[j]][2] == True):
                    break
                current[word[j]][1] += 1
            else:
                id = id+1
            current = current.setdefault(word[j], [{},1, j == 0, id, j == (len(word)-1)])[0]
    return trie

def search(trie, word):
        current = trie[0]
        numberOfEqualLetters =  0
        j = len(word)-1
        suffix = ''
        while(word[j] in current):
            suffix += word[j]
            numberOfEqualLetters += 1
            currentNoOfWordsPassThrough = current.get(word[j])[1]
            current = current.get(word[j])[0]
            j -= 1
        if(numberOfEqualLetters > 0):
            return [numberOfEqualLetters, currentNoOfWordsPassThrough]
        else:
            return [] #trie doesn't have suffix
def printify(trie):
    for k in trie[0]:
        if(((trie[0])[k])[4] == True):
            dot.edge('0',str(((trie[0])[k])[3]))
        dot.node(str(((trie[0])[k])[3]), k+' |'+str(((trie[0])[k])[1])+'| '+str(((trie[0])[k])[2]))
        for j in ((trie[0])[k])[0]:
            dot.edge(str(((trie[0])[k])[3]),str(((((trie[0])[k])[0])[j])[3]))
        printify((trie[0])[k])

def seperate(trainsize):
    db = HmlDB("..//hml.db")
    seed, triple, br, noun_train, test = 432, {}, 0, [], []
    adjective_train, verb_train, adverb_train, pronoun_train, numeral_train = [], [], [], [], []
    allItems = HmlDB.select_all(db)
    random.seed(seed)
    random.shuffle(allItems)

    for item in allItems:
        if not item[0] in triple:
            triple[item[0]] = [[item[1],item[2]]]
        else:
            triple[item[0]]+=[[item[1],item[2]]]
    for lemma in triple:
        br+=len(triple[lemma])
        if(br>len(allItems)*trainsize):
            test+=[i[0] for i in triple[lemma]]
        else:
            for j in triple[lemma]:
                if j[1][0] == 'N':
                    noun_train += [j[0]]
                if j[1][0] == 'A':
                    adjective_train += [j[0]]
                if j[1][0] == 'V':
                    verb_train += [j[0]]
                if j[1][0] == 'R':
                    adverb_train += [j[0]]
                if j[1][0] == 'P':
                    pronoun_train += [j[0]]
                if j[1][0] == 'M':
                    numeral_train += [j[0]]
    set(test)
    set(noun_train)
    set(adjective_train)
    set(verb_train)
    set(adverb_train)
    set(pronoun_train)
    set(numeral_train)
    write('../test.pickle',test)
    write('../trainNounTrie.pickle',put([{}],noun_train))
    write('../trainAdjectiveTrie.pickle',put([{}],adjective_train))
    write('../trainVerbTrie.pickle',put([{}],verb_train))
    write('../trainAdverbTrie.pickle',put([{}],adverb_train))
    write('../trainPronounTrie.pickle',put([{}],pronoun_train))
    write('../trainNumeralTrie.pickle',put([{}],numeral_train))

def decideFromTrainTries():
    nounTrainTrie=read('../trainNounTrie.pickle') #dict
    testTrie=read('../test.pickle') #list
    nounTrieResult=search(nounTrainTrie,testTrie[0])
    print(nounTrieResult,testTrie[0])


dot = Digraph()
dot.node('0', 'ROOT')
dot.format = 'svg'
trainsize = 0.8
seperate(trainsize)
decideFromTrainTries()



#dot.render("..//trieDot.gv", view=True)


