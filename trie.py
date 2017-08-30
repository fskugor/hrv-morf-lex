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


def put(trie, words): #this function creates trie from list of words
    id = 0
    for word in words:
        current = trie[0] # [0] we set current position in trie as [0] because we are starting from root
        numberOfEqualLetters = 0
        for j in range(len(word)-1, -1, -1): # every letter is node, reverse order
            if word[j] in current:
                numberOfEqualLetters += 1
                if((numberOfEqualLetters == len(word)) and current[word[j]][2] == True):
                    break
                current[word[j]][1] += 1
            else:
                id = id+1
            current = current.setdefault(word[j], [{},1, j == 0, id, j == (len(word)-1)])[0]#key is letter,
    return trie # values are empty dict(for storing next letter of word), number of other words containing that
                # node(letter), info if the node is first letter of word, id for visualization, if it's last letter(used for visualization)
def search(trie, word):
        current = trie[0]
        numberOfEqualLetters =  0
        lenght=len(word)
        j = len(word)-1
        while(word[j] in current):
            numberOfEqualLetters += 1
            currentNoOfWordsPassThrough = current.get(word[j])[1]
            current = current.get(word[j])[0]
            if lenght == 2:
                if j == 0: break
            elif j == 1: break
            j -= 1
        if(numberOfEqualLetters > 1):
            return [numberOfEqualLetters, currentNoOfWordsPassThrough]#returns suffix length and number of words
        else:                                                         #which are containing the same suffix
            return [0, 0] #trie doesn't have suffix that's long enough for considering
def printify(trie):
    for k in trie[0]: #recursive function for adding nodes and connect them to edges for visualization to .dot file
        if(((trie[0])[k])[4] == True):
            dot.edge('0',str(((trie[0])[k])[3])) #connect node to root
        dot.node(str(((trie[0])[k])[3]), k+' |'+str(((trie[0])[k])[1])+'| '+str(((trie[0])[k])[2]))#describe node
        for j in ((trie[0])[k])[0]:
            dot.edge(str(((trie[0])[k])[3]),str(((((trie[0])[k])[0])[j])[3])) #connect node with next node
        printify((trie[0])[k]) #go deeper, for to the end of word

def separate(trainsize):
    print("Enter separate()")
    db = HmlDB("..//hml.db")
    seed, triple, br, noun_train, test = 4, {}, 0, [], []
    adjective_train, verb_train, adverb_train, pronoun_train, numeral_train = [], [], [], [], []
    allItems = HmlDB.select_all(db) # get all triples from db
    random.seed(seed)
    random.shuffle(allItems)

    for item in allItems:
        if not item[0] in triple:
            triple[item[0]] = [[item[1],item[2]]] # add lemma as key, other as value
        else:                                     # for easier separating train from test corpus
            triple[item[0]]+=[[item[1],item[2]]]  # as it cannot be the same lemma in both corpus
    for lemma in triple:
        br+=len(triple[lemma])
        if(br>len(allItems)*trainsize):
            test+=[i[0] for i in triple[lemma]]
        else:
            for j in triple[lemma]: #separate N trie, A trie ...--->to 6 train tries
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
    testToset=set(test)
    write('../test.pickle', list(testToset))  # write all to files,
    write('../trainNounTrie.pickle', put([{}], set(noun_train)))
    write('../trainAdjectiveTrie.pickle', put([{}], set(adjective_train)))
    write('../trainVerbTrie.pickle', put([{}], set(verb_train)))
    write('../trainAdverbTrie.pickle', put([{}], set(adverb_train)))
    write('../trainPronounTrie.pickle', put([{}], set(pronoun_train)))
    write('../trainNumeralTrie.pickle', put([{}], set(numeral_train)))
    write('../allTriplesDict.picle',triple)
    print("Leaving separate()")

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
    print(len(testTrie))
    # br=0
    for testWord in testTrie:
        # br+=1
        # if br==101: break
        if len(testWord) > 1:
            msd = HmlDB.select_by_token(db,testWord)
            nounResultFromSearchTrie = search(nounTrainTrie, testWord)
            adjResultFromSearchTrie = search(adjectiveTrainTrie, testWord)
            verbResultFromSearchTrie = search(verbTrainTrie, testWord)
            advResultFromSearchTrie = search(adverbTrainTrie, testWord)
            pronResultFromSearchTrie = search(pronounTrainTrie, testWord)
            numResultFromSearchTrie = search(numeralTrainTrie, testWord)
            #separate result from search method so we can decide which type of word is our test word
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
            #find max suffix(N,V,A,M,P or R) and max number of words passing through this suffix
            if maxSuffix == nounResultFromSearchTrie[0]: resultMax2 += [['N', suffixLenghts[0], wordsWithEqualSuffix[0]]]
            if maxSuffix == verbResultFromSearchTrie[0]: resultMax2 += [['V', suffixLenghts[1], wordsWithEqualSuffix[1]]]
            if maxSuffix == adjResultFromSearchTrie[0]: resultMax2 += [['A', suffixLenghts[3], wordsWithEqualSuffix[3]]]
            if maxSuffix == numResultFromSearchTrie[0]: resultMax2 += [['M', suffixLenghts[5], wordsWithEqualSuffix[5]]]
            if maxSuffix == pronResultFromSearchTrie[0]: resultMax2 += [['P', suffixLenghts[4], wordsWithEqualSuffix[4]]]
            if maxSuffix == advResultFromSearchTrie[0]: resultMax2 += [['R', suffixLenghts[2], wordsWithEqualSuffix[2]]]

            resultMax12=resultMax1+resultMax2
            found=False
            for i in range(len(resultMax12)-1):
                for j in range(len(resultMax12)):
                    if resultMax12[i]!=resultMax12[j]:
                        if resultMax12[i][1]==maxSuffix and resultMax12[i][2]==maxNoOfWords:
                            finalRes=resultMax12[i]
                            found=True
                            break
                        if resultMax12[i][1]>=resultMax12[j][1] and resultMax12[i][2]>1:
                            finalRes=resultMax12[i]
                        elif resultMax12[i][1]<=resultMax12[j][1] and resultMax12[j][2]>1:
                            finalRes=resultMax12[j]
                        else:
                            if resultMax12[i][1]==maxSuffix:
                                finalRes=resultMax12[i]
                            elif resultMax12[j][1]==maxSuffix:
                                finalRes=resultMax12[j]
                    else:
                        finalRes=resultMax12[i]
                if found: break

            # print(testWord)
            # print(resultMax12)
            # print(finalRes)
            if finalRes[0] == msd:
                pogodija+=1
                # print("YES")
            else:
                falija+=1
                # print("NO")
            k="pogodija: "+str(pogodija)+ "\nfalija: "+str(falija)
            with open('guessOrfail.txt', 'w') as outfile:
                json.dump(k, outfile)




dot = Digraph()
dot.node('0', 'ROOT')
dot.format = 'svg'
trainsize = 0.9
separate(trainsize)
t=time.time()
decideFromTrainTries()
print(time.time()-t)

#dot.render("..//trieDot.gv", view=True)


