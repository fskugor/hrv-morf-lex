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

def write(filename, l_ist):
    with open(filename, 'wb') as f:
        pickle.dump(put([{}],l_ist), f, pickle.HIGHEST_PROTOCOL)


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
            current = current.get(word[j])[0]
            j -= 1
        if(numberOfEqualLetters > 0):
            print("Trie has suffix for word '"+word+"'. Suffix: '"+suffix+"'.")
        else:
            print("Trie doesn't have suffix for word '"+word+"'.")
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
    seed = 432
    noun = [i[0] for i in  HmlDB.select_token_by_msd(db,'N%')]
    adjective= [i[0] for i in  HmlDB.select_token_by_msd(db,'A%')]
    verb = [i[0] for i in  HmlDB.select_token_by_msd(db,'V%')]
    adverb = [i[0] for i in  HmlDB.select_token_by_msd(db,'R%')] #prilog
    pronoun = [i[0] for i in  HmlDB.select_token_by_msd(db,'P%')] #zamjenica
    numeral = [i[0] for i in  HmlDB.select_token_by_msd(db,'M%')]
    random.seed(seed)
    random.shuffle(noun)
    noun_train,noun_test = noun[:int(len(noun)*trainsize)], noun[int(len(noun)*trainsize):]
    random.shuffle(adjective)
    adjective_train, adjective_test = adjective[:int(len(adjective)*trainsize)], adjective[int(len(adjective)*trainsize):]
    random.shuffle(verb)
    verb_train, verb_test = verb[:int(len(verb)*trainsize)], verb[int(len(verb)*trainsize):]
    random.shuffle(adverb)
    adverb_train, adverb_test = adverb[:int(len(adverb)*trainsize)], adverb[int(len(adverb)*trainsize):]
    random.shuffle(pronoun)
    pronoun_train, pronoun_test = pronoun[:int(len(pronoun)*trainsize)], pronoun[int(len(pronoun)*trainsize):]
    random.shuffle(numeral)
    numeral_train, numeral_test = numeral[:int(len(numeral)*trainsize)], numeral[int(len(numeral)*trainsize):]
    test = noun_test + adjective_test + verb_test + adverb_test + pronoun_test + numeral_test
    train = noun + adjective + verb + adverb + pronoun + numeral
    set(test)
    print("Len Test",len(test))
    print("Len Train",len(train))
    write('../test.pickle',test)
    write('../trainNounTrie.pickle',noun_train)
    write('../trainAdjectiveTrie.pickle',adjective_train)
    write('../trainVerbTrie.pickle',verb_train)
    write('../trainAdverbTrie.pickle',adverb_train)
    write('../trainPronounTrie.pickle',pronoun_train)
    write('../trainNumeralTrie.pickle',numeral_train)




dot = Digraph()
dot.node('0', 'ROOT')
dot.format = 'svg'
trainsize = 0.8
seperate(trainsize)


#dot.render("..//trieDot.gv", view=True)


