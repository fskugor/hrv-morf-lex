import os
import json
from pprint import pprint
from graphviz import Digraph
from hmldb import HmlDB
import random
import pickle
import time


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
            if j == 0: break
            j -= 1
        if(numberOfEqualLetters > 1):
            return [numberOfEqualLetters, currentNoOfWordsPassThrough]#returns suffix length and number of words
        else:                                                         #which are containing the same suffix
            return [0, 0] #trie doesn't have suffix that's long enough for considering
def printify(trie):
    for k in trie[0]:
        if(((trie[0])[k])[4] == True):
            dot.edge('0',str(((trie[0])[k])[3]))
        dot.node(str(((trie[0])[k])[3]), k+' |'+str(((trie[0])[k])[1])+'| '+str(((trie[0])[k])[2]))
        for j in ((trie[0])[k])[0]:
            dot.edge(str(((trie[0])[k])[3]),str(((((trie[0])[k])[0])[j])[3]))
        printify((trie[0])[k])

def seperate(trainsize):
    print("Enter seperate()")
    db = HmlDB("..//hml.db")
    seed, triple, br, noun_train, test = 432, {}, 0, [], []
    adjective_train, verb_train, adverb_train, pronoun_train, numeral_train = [], [], [], [], []
    allItems = HmlDB.select_all(db)
    random.seed(seed)
    random.shuffle(allItems)
    print("sve rici ",len(allItems))
    print("trening ", len(allItems)*trainsize)
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

    write('../test.pickle', set(test))
    write('../trainNounTrie.pickle', put([{}], set(noun_train)))
    write('../trainAdjectiveTrie.pickle', put([{}], set(adjective_train)))
    write('../trainVerbTrie.pickle', put([{}], set(verb_train)))
    write('../trainAdverbTrie.pickle', put([{}], set(adverb_train)))
    write('../trainPronounTrie.pickle', put([{}], set(pronoun_train)))
    write('../trainNumeralTrie.pickle', put([{}], set(numeral_train)))
    write('../allTriplesDict.picle',triple)
    print("Leaving seperate()")

def decideFromTrainTries():
    db = HmlDB("..//hml.db")
    print("Enter decideFromTrainTries()")
    nounTrainTrie = read('../trainNounTrie.pickle') #dictionaries
    adjectiveTrainTrie = read('../trainAdjectiveTrie.pickle')
    verbTrainTrie = read('../trainVerbTrie.pickle')
    adverbTrainTrie = read('../trainAdverbTrie.pickle')
    pronounTrainTrie = read('../trainPronounTrie.pickle')
    numeralTrainTrie = read('../trainNumeralTrie.pickle')
    pogodija = 0
    falija = 0
    allTriples=read('../allTriplesDict.picle')
    testTrie = read('../test.pickle') #list
    #print(nounTrieResult,testTrie[0])
    for testWord in testTrie:
        msd = HmlDB.select_by_token(db,testWord)
        nounResultFromSearchTrie = search(nounTrainTrie, testWord)
        adjResultFromSearchTrie = search(adjectiveTrainTrie, testWord)
        verbResultFromSearchTrie = search(verbTrainTrie, testWord)
        advResultFromSearchTrie = search(adverbTrainTrie, testWord)
        pronResultFromSearchTrie = search(pronounTrainTrie, testWord)
        numResultFromSearchTrie = search(numeralTrainTrie, testWord)
        # print(testWord)
        # print("nounResultFromSearchTrie: ", nounResultFromSearchTrie)
        # print("verbResultFromSearchTrie: ", verbResultFromSearchTrie)
        # print("advResultFromSearchTrie: ", advResultFromSearchTrie)
        # print("adjResultFromSearchTrie: ", adjResultFromSearchTrie)
        # print("pronResultFromSearchTrie: ", pronResultFromSearchTrie)
        # print("numResultFromSearchTrie: ", numResultFromSearchTrie)

        suffixLenghts = [nounResultFromSearchTrie[0], verbResultFromSearchTrie[0], \
                         advResultFromSearchTrie[0], adjResultFromSearchTrie[0], \
                         pronResultFromSearchTrie[0], numResultFromSearchTrie[0]]
        wordsWithEqualSuffix = [nounResultFromSearchTrie[1], verbResultFromSearchTrie[1], \
                                advResultFromSearchTrie[1], adjResultFromSearchTrie[1], \
                                pronResultFromSearchTrie[1], numResultFromSearchTrie[1]]
        maxNoOfWords = max(wordsWithEqualSuffix)
        maxSuffix = max(suffixLenghts)
        resultMax1, resultMax2, resultMax12, finalRes = [], [], [], []
        if maxNoOfWords == nounResultFromSearchTrie[1]: resultMax1 += [['N', suffixLenghts[0], wordsWithEqualSuffix[0]]]
        if maxNoOfWords == verbResultFromSearchTrie[1]: resultMax1 += [['V', suffixLenghts[1], wordsWithEqualSuffix[1]]]
        if maxNoOfWords == adjResultFromSearchTrie[1]: resultMax1 += [['A', suffixLenghts[3], wordsWithEqualSuffix[3]]]
        if maxNoOfWords == numResultFromSearchTrie[1]: resultMax1 += [['M', suffixLenghts[5], wordsWithEqualSuffix[5]]]
        if maxNoOfWords == pronResultFromSearchTrie[1]: resultMax1 += [['P', suffixLenghts[4], wordsWithEqualSuffix[4]]]
        if maxNoOfWords == advResultFromSearchTrie[1]: resultMax1 += [['R', suffixLenghts[2], wordsWithEqualSuffix[2]]]

        if maxSuffix == nounResultFromSearchTrie[0]: resultMax2 += [['N', suffixLenghts[0], wordsWithEqualSuffix[0]]]
        if maxSuffix == verbResultFromSearchTrie[0]: resultMax2 += [['V', suffixLenghts[1], wordsWithEqualSuffix[1]]]
        if maxSuffix == adjResultFromSearchTrie[0]: resultMax2 += [['A', suffixLenghts[3], wordsWithEqualSuffix[3]]]
        if maxSuffix == numResultFromSearchTrie[0]: resultMax2 += [['M', suffixLenghts[5], wordsWithEqualSuffix[5]]]
        if maxSuffix == pronResultFromSearchTrie[0]: resultMax2 += [['P', suffixLenghts[4], wordsWithEqualSuffix[4]]]
        if maxSuffix == advResultFromSearchTrie[0]: resultMax2 += [['R', suffixLenghts[2], wordsWithEqualSuffix[2]]]

        if(resultMax1[0] not in  resultMax2): #if we cannot decide beetwean results, than observe just these two and decide
            resultMax12 = resultMax1+resultMax2
            for triple in resultMax12:
                if(triple[1] == maxSuffix and triple[2] == maxNoOfWords):
                    finalRes = triple
                elif triple[1] == maxSuffix:
                    finalRes = triple
        else:
            if len(resultMax1) == 1:
                finalRes = resultMax1[0]
            else:
                if resultMax1[0][1]>resultMax1[1][1]:
                    finalRes = resultMax1[0]
                else:
                    finalRes = resultMax1[1]
        found = False
        if finalRes[0] == msd:
            pogodija+=1
        else:
            falija+=1
        k="pogodija: "+str(pogodija)+ "\nfalija: "+str(falija)
        with open('guessOrfail.txt', 'w') as outfile:
            json.dump(k, outfile)




dot = Digraph()
dot.node('0', 'ROOT')
dot.format = 'svg'
trainsize = 0.87
seperate(trainsize)
t=time.time()
decideFromTrainTries()
print(time.time()-t)

#dot.render("..//trieDot.gv", view=True)


