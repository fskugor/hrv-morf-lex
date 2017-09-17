import os
import json
from pprint import pprint
from graphviz import Digraph
from hmldb import HmlDB
import random
import pickle
import time

noun, adj, ver, pron, num = 0, 0, 0, 0, 0
nounSaidAdj, nounSaidVerb, nounSaidPron, nounSaidNum = 0, 0, 0, 0
adjSaidNoun, adjSaidVerb, adjSaidPron, adjSaidNum = 0, 0, 0, 0
verbSaidNoun, verbSaidPron, verbSaidAdj, verbSaidNum = 0, 0, 0, 0
pronSaidNoun, pronSaidAdj, pronSaidNum, pronSaidVerb = 0, 0, 0, 0
numSaidNoun, numSaidAdj, numSaidVerb, numSaidPron = 0, 0, 0, 0
nounSaidUnknown, pronSaidUnknown, adjSaidUnknown, verbSaidUnknown, numSaidUnknown, unknown = 0, 0, 0, 0, 0, 0

def read(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def write(filename, trie):
    with open(filename, 'wb') as f:
        pickle.dump(trie, f, pickle.HIGHEST_PROTOCOL)


def put(trie, words): #this function creates trie from list of words
    id = 0
    for key in words:
        wordsList=[i[0] for i in words[key]]
        wordsSet=set(wordsList) #avoid adding duplicate words
        for word in wordsSet:# word is pair:token,msd, we need only token
            current = trie[0] # [0] we set current position in trie as [0] because we are starting from root
            for j in range(len(word)-1, -1, -1): # every letter is node, reverse order
                if word[j] in current:
                    current[word[j]][1] += 1
                else:
                    id = id+1
                if j == 0:
                    current = current.setdefault(word[j], [{},1, j == 0, id, j == (len(word)-1)])[0]#key is letter,
                else:
                    current = current.setdefault(word[j], [{},1, j == 0, id, j == (len(word)-1)])[0]
    return trie # values are empty dict(for storing next letter of word), number of other words containing that
                # node(letter), info if the node is first letter of word, id for visualization, if it's last letter(used for visualization)
def searchSuperWord(trie,suffix):
    tempList = []
    for key in trie:
        if trie[key][2]:
            suffix+=key
            return suffix[::-1]
        else:
            tempList+=[trie[key][1]]
    if tempList:
        maxWord=max(tempList)
        for key in trie:
            if trie[key][1] == maxWord:
                suffix+=key
                return searchSuperWord(trie.get(key)[0],suffix)


def search(trie, word):
        current = trie[0]
        j = len(word)-1
        subword=''
        suffix=''
        superword=''
        currentNoOfWordsPassThrough=0
        while(word[j] in current):
            suffix+=word[j]
            if current.get(word[j])[2] and j!=0:
                subword = suffix[::-1]
            currentNoOfWordsPassThrough = current.get(word[j])[1]
            current = current.get(word[j])[0]
            if j == 0:
                superword = searchSuperWord(current, suffix)
                break
            j -= 1
        if superword:
            return [superword,False, True]
        elif subword:
            return [subword,True, False]
        return [suffix[::-1],False, False]# returns either suffix or subword(if exists)
def printify(trie):
    for k in trie[0]: #recursive function for adding nodes and connect them to edges for visualization to .dot file
        if(((trie[0])[k])[4] == True):
            dot.edge('0',str(((trie[0])[k])[3])) #connect node to root
        dot.node(str(((trie[0])[k])[3]), k+' |'+str(((trie[0])[k])[1])+'| '+str(((trie[0])[k])[2]))#describe node
        for j in ((trie[0])[k])[0]:
            dot.edge(str(((trie[0])[k])[3]),str(((((trie[0])[k])[0])[j])[3])) #connect node with next node
        printify((trie[0])[k])

def separate(trainsize):
    #print("Enter separate()")
    db = HmlDB("../hml.db")
    seed, triple, br, noun_train, test = 125, {}, 0, {}, {}
    adjective_train, verb_train, pronoun_train, numeral_train = {}, {}, {}, {}
    adjective, verb, noun, pronoun, numeral = {}, {}, {}, {}, {}
    allItems = HmlDB.select_all(db) # get all triples from db
    random.seed(seed)
    random.shuffle(allItems)
    nouns_count=HmlDB.count_lemmas(db,"N%")
    adj_count=HmlDB.count_lemmas(db,"A%")
    verb_count=HmlDB.count_lemmas(db,"V%")
    pron_count=HmlDB.count_lemmas(db,"P%")
    num_count=HmlDB.count_lemmas(db,"M%")
    for item in allItems:
        if not item[0] in triple:
            triple[item[0]] = [[item[1],item[2]]] # add lemma as key, other as value
        else:                                     # for easier separating train from test corpus
            triple[item[0]]+=[[item[1],item[2]]]  # as it cannot be the same lemma in both corpus
    for lemma in triple:
        for j in triple[lemma]:
            if j[1][0] == 'N':
                if not lemma in noun: noun[lemma] = [(j[0],j[1])]
                else: noun[lemma] += [(j[0],j[1])]
            if j[1][0] == 'A':
                if not lemma in adjective: adjective[lemma] = [(j[0],j[1])]
                else: adjective[lemma] += [(j[0],j[1])]
            if j[1][0] == 'V':
                if not lemma in verb: verb[lemma] = [(j[0],j[1])]
                else: verb[lemma] += [(j[0],j[1])]
            if j[1][0] == 'P':
                if not lemma in pronoun: pronoun[lemma] = [(j[0],j[1])]
                else: pronoun[lemma] += [(j[0],j[1])]
            if j[1][0] == 'M':
                if not lemma in numeral: numeral[lemma] = [(j[0],j[1])]
                else: numeral[lemma] += [(j[0],j[1])]
    for key in noun:
        br+=1
        for j in noun[key]:
            if br>(nouns_count*trainsize):
                if not key in test: test[key] = [(j[0],j[1])]
                else: test[key] += [(j[0],j[1])]
            else:
                if not key in noun_train: noun_train[key] = [(j[0],j[1])]
                else: noun_train[key] += [(j[0],j[1])]
    br = 0
    for key in adjective:
        br+=1
        for j in adjective[key]:
            if br>(adj_count*trainsize):
                if not key in test: test[key] = [(j[0],j[1])]
                else: test[key] += [(j[0],j[1])]
            else:
                if not key in adjective_train: adjective_train[key] = [(j[0],j[1])]
                else: adjective_train[key] += [(j[0],j[1])]
    br = 0
    for key in verb:
        br+=1
        for j in verb[key]:
            if br>(verb_count*trainsize):
                if not key in test: test[key] = [(j[0],j[1])]
                else: test[key] += [(j[0],j[1])]
            else:
                if not key in verb_train: verb_train[key] = [(j[0],j[1])]
                else: verb_train[key] += [(j[0],j[1])]
    br = 0
    for key in pronoun:
        br+=1
        for j in pronoun[key]:
            if br>(pron_count*0.5):
                if not key in test: test[key] = [(j[0],j[1])]
                else: test[key] += [(j[0],j[1])]
            else:
                if not key in pronoun_train: pronoun_train[key] = [(j[0],j[1])]
                else: pronoun_train[key] += [(j[0],j[1])]
    br = 0
    for key in numeral:
        br+=1
        for j in numeral[key]:
            if br>(num_count*0.5):
                if not key in test: test[key] = [(j[0],j[1])]
                else: test[key] += [(j[0],j[1])]
            else:
                if not key in numeral_train: numeral_train[key] = [(j[0],j[1])]
                else: numeral_train[key] += [(j[0],j[1])]

    write('../test.pickle', test)  # write all to files,
    write('../trainNounTrie.pickle', put([{}], noun_train))
    write('../trainAdjectiveTrie.pickle', put([{}], adjective_train))
    write('../trainVerbTrie.pickle', put([{}], verb_train))
    write('../trainPronounTrie.pickle', put([{}], pronoun_train))
    write('../trainNumeralTrie.pickle', put([{}], numeral_train))
    write('../trainNounDictionary.picle',noun_train)
    write('../trainAdjectiveDictionary.picle',adjective_train)
    write('../trainVerbDictionary.picle',verb_train)
    write('../trainPronounDictionary.picle',pronoun_train)
    write('../trainNumeralDictionary.picle',numeral_train)
    #print("Leaving separate()")

def confusionMatrix(fail):
    global nounSaidAdj, nounSaidVerb, nounSaidPron, nounSaidNum
    global adjSaidNoun, adjSaidVerb, adjSaidPron, adjSaidNum
    global verbSaidNoun, verbSaidPron, verbSaidAdj, verbSaidNum
    global pronSaidNoun, pronSaidAdj, pronSaidNum, pronSaidVerb
    global numSaidNoun, numSaidAdj, numSaidVerb, numSaidPron
    global nounSaidUnknown, pronSaidUnknown, adjSaidUnknown, verbSaidUnknown, numSaidUnknown, unknown
    if len(fail)==2: #fail is string, for example NA, N is actual type of test word, A is prediction
        if fail[0] == 'N':
            if fail[1] == 'A': nounSaidAdj+=1
            elif fail[1] == 'V': nounSaidVerb+=1
            elif fail[1] == 'P': nounSaidPron+=1
            elif fail[1] == 'M': nounSaidNum+=1
        elif fail[0] == 'A':
            if fail[1] == 'N': adjSaidNoun+=1
            elif fail[1] == 'V': adjSaidVerb+=1
            elif fail[1] == 'P': adjSaidPron+=1
            elif fail[1] == 'M': adjSaidNum+=1
        elif fail[0] == 'V':
            if fail[1] == 'N': verbSaidNoun+=1
            elif fail[1] == 'A': verbSaidAdj+=1
            elif fail[1] == 'P': verbSaidPron+=1
            elif fail[1] == 'M': verbSaidNum+=1
        elif fail[0] == 'P':
            if fail[1] == 'N': pronSaidNoun+=1
            elif fail[1] == 'A': pronSaidAdj+=1
            elif fail[1] == 'V': pronSaidVerb+=1
            elif fail[1] == 'M': pronSaidNum+=1
        elif fail[0] == 'M':
            if fail[1] == 'N': numSaidNoun+=1
            elif fail[1] == 'A': numSaidAdj+=1
            elif fail[1] == 'V': numSaidVerb+=1
            elif fail[1] == 'P': numSaidPron+=1
    else:
        if fail == '' : unknown+=1
        elif fail == 'N': nounSaidUnknown+=1
        elif fail == 'A': adjSaidUnknown+=1
        elif fail == 'V': verbSaidUnknown+=1
        elif fail == 'P': pronSaidUnknown+=1
        elif fail == 'M': numSaidUnknown+=1

def compare(original, generated):
    originalGroupByToken, generatedGroupByToken, br, guessedMsd, generatedKeys, originalKeys = {}, {}, 0, 0, [], []
    global noun, adj, ver, pron, num

    for pair in original:
        if not pair[0] in originalGroupByToken: originalGroupByToken[pair[0]]=[pair[1]]
        else: originalGroupByToken[pair[0]]+=[pair[1]]
    for pair in generated:
        if not pair[0] in generatedGroupByToken: generatedGroupByToken[pair[0]]=[pair[1]]
        else: generatedGroupByToken[pair[0]]+=[pair[1]]
    generatedKeys = [key for key in generatedGroupByToken]
    originalKeys = [key for key in originalGroupByToken]
    if set(generatedKeys) == set(originalKeys) and original[0][1][0]==generated[0][1][0]:
        #print(generatedGroupByToken)
        #print('++++++++++++++++++++++++++++++++++++++++++++')
        #print(originalGroupByToken)
        if generated[0][1][0] == 'N': noun+=1
        elif generated[0][1][0] == 'A': adj+=1
        elif generated[0][1][0] == 'V': ver+=1
        elif generated[0][1][0] == 'P': pron+=1
        elif generated[0][1][0] == 'M': num+=1
        for key in originalGroupByToken:
            for msd in originalGroupByToken[key]:
                if msd in generatedGroupByToken[key]:
                    br+=1
                    break
        guessedMsd=br/len(originalGroupByToken)
        f = open('msd.txt', 'a')
        f.write(str(guessedMsd)+', '+original[0][1][0]+'\n')
        return [True]
    elif set(generatedKeys) == set(originalKeys) and original[0][1][0]!=generated[0][1][0]:
        return [False, original[0][1][0], generated[0][1][0]] #for confusion matrix
    else:
        return[False,original[0][1][0]] #also for confusion matrix

def classifySubword(resultFromSearchTrie, trainDict, testWord, testPairs):
    found, lemma, token = False, '', ''
    pairs_train = lambda lemma: trainDict[lemma] if lemma else []
    for i in trainDict:
        for j in trainDict[i]:
            if j[0] == resultFromSearchTrie:
                lemma = i  # lemma tj oblikX koji je najsličniji našem obliku(tokenu=lemma) tj riječi koju testiramo
                found=True
                break
        if found: break

    if pairs_train(lemma):
        #print("Lemma za generiranje testnog prikaza: ",lemma)
        #print("\nTrain token: ",resultFromSearchTrie)
        #print("\nTestni token: ",testWord)
        generatedTestModel, testModelOrigin = [], []
        pairs = pairs_train(lemma)
        subword = len(resultFromSearchTrie)
        preffixTest = testWord[:-subword]
        for i in pairs: #zamijenimo svaki prefix od treninga sa našim od testa
            generatedTestModel += [(preffixTest.lower()+i[0],i[1])]
        for j in testPairs:
            testModelOrigin +=[(j[0].lower(),j[1])]
        result = compare(testModelOrigin,generatedTestModel)
        if result[0]:
            return [True]
            #print("Test model found: ",generatedTestModel)
        else:
            if len(result)==3:
                return [False, result[1], result[2]]
            else:
                return [False, result[1]]
            #print("Test model not found")
    else: return [False]

def classifySuperword(resultFromSearchTrie, trainDict, testWord, testPairs):
    found, lemma, token = False, '', ''
    pairs_train = lambda lemma: trainDict[lemma] if lemma else []
    for i in trainDict:
        for j in trainDict[i]:
            if j[0] == resultFromSearchTrie:
                lemma = i  # lemma tj oblikX koji je najsličniji našem obliku(tokenu=lemma) tj riječi koju testiramo
                found=True
                break
        if found: break

    if pairs_train(lemma):
        #print("Lemma za generiranje testnog prikaza: ",lemma)
        #print("\nTrain token: ",lemma)
        #print("\nTestni token: ",testWord)
        generatedTestModel, testModelOrigin = [], []
        pairs = pairs_train(lemma)
        test=len(testWord)
        preffixTrain = resultFromSearchTrie[:-test]
        #print(preffixTrain)
        for i in pairs: #zamijenimo svaki prefix od treninga sa našim od testa
            generatedTestModel += [(i[0][len(preffixTrain):],i[1])]
        for j in testPairs:
            testModelOrigin +=[(j[0].lower(),j[1])]
        result = compare(testModelOrigin,generatedTestModel)
        if result[0]:
            return [True]
            #print("Test model found: ",generatedTestModel)
        else:
            if len(result)==3:
                return [False, result[1], result[2]]
            else:
                return [False, result[1]]
            #print("Test model not found")
    else: return [False]

def classifySuffix(resultFromSearchTrie, trainDict, testWord, testPairs):
    found, lemma, token = False, '', ''
    pairs_train = lambda lemma: trainDict[lemma] if lemma else []
    if resultFromSearchTrie: #ako postoji suffix od testWOrd u treningu nađi lemmu od tog oblika
        for i in trainDict:
            for j in trainDict[i]:
                if j[0][-len(resultFromSearchTrie):] == resultFromSearchTrie:
                    lemma = i  # lemma tj oblikX koji je najsličniji našem obliku(tokenu=lemma) tj riječi koju testiramo
                    token = j[0]
                    found=True
                    break
            if found: break

    if pairs_train(lemma):
        #print("Lemma za generiranje testnog prikaza: ",lemma)
        #print("\nTrain token: ",token)
        #print("\nTestni token: ",testWord)
        lemmaEqSuffix = False
        generatedTestModel, testModelOrigin = [], []
        pairs = pairs_train(lemma)
        suffix = len(resultFromSearchTrie)
        if token == resultFromSearchTrie:
            preffixTrain = token[-suffix:]
            lemmaEqSuffix = True
        else: preffixTrain = token[:-suffix]
        preffixTest = testWord[:-suffix]
        #print(preffixTrain)
        #print(preffixTest)
        for i in pairs: #zamijenimo svaki prefix od treninga sa našim od testa
            if lemmaEqSuffix:
                generatedTestModel += [(preffixTest.lower()+i[0],i[1])]
            else:
                generatedTestModel += [(preffixTest.lower()+i[0][len(preffixTrain):].lower(),i[1])]
        for j in testPairs:
            testModelOrigin +=[(j[0].lower(),j[1])]
        result = compare(testModelOrigin,generatedTestModel)
        if result[0]:
            return [True]
            #print("Test model found: ",generatedTestModel)
        else:
            if len(result)==3:
                return [False, result[1], result[2]]
            else:
                return [False, result[1]]
            #print("Test model not found")
    else: return [False]

def decideFromTrainTries():
    db = HmlDB("..//hml.db")
    #print("Enter decideFromTrainTries()")
    nounTrainTrie = read('../trainNounTrie.pickle') #nested dictionaries -->tries
    adjectiveTrainTrie = read('../trainAdjectiveTrie.pickle')
    verbTrainTrie = read('../trainVerbTrie.pickle')
    adverbTrainTrie = read('../trainAdverbTrie.pickle')
    pronounTrainTrie = read('../trainPronounTrie.pickle')
    numeralTrainTrie = read('../trainNumeralTrie.pickle')
    noun_train = read('../trainNounDictionary.picle') #dictionaries
    adjective_train = read('../trainAdjectiveDictionary.picle')
    verb_train = read('../trainVerbDictionary.picle')
    pronoun_train = read('../trainPronounDictionary.picle')
    numeral_train = read('../trainNumeralDictionary.picle')
    pogodija, falija = 0, 0
    print("Pridjevi: ",len(adjective_train))
    print("Zamjenice: ",len(pronoun_train))
    print("Glagoli: ",len(verb_train))
    print("Imenice: ",len(noun_train))
    print("Brojevi: ",len(numeral_train))
    testDict = read('../test.pickle') #list
    for key in testDict: # prolazimo kroz svaki prikaz i uzmemo oblik(token=lemma) za testiranje
        testWord=testDict[key][0][0] #oblik koji nije nužno jednak lemmi
        nounResultFromSearchTrie = search(nounTrainTrie, testWord)
        adjResultFromSearchTrie = search(adjectiveTrainTrie, testWord)
        verbResultFromSearchTrie = search(verbTrainTrie, testWord)
        pronResultFromSearchTrie = search(pronounTrainTrie, testWord)
        numResultFromSearchTrie = search(numeralTrainTrie, testWord)
        #print(testWord)

        found = False
        halfFalse, fullFalse = '', ''
        if nounResultFromSearchTrie[1]:
            ##print("Subword from noun trie: ",nounResultFromSearchTrie)
            resultFromClassify = classifySubword(nounResultFromSearchTrie[0], noun_train, testWord, testDict[key])
            if resultFromClassify[0]:
                found = True
                ##print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            else:
                if len(resultFromClassify)==3:
                    halfFalse = resultFromClassify[1] + resultFromClassify[2]
                elif len(resultFromClassify)==2:
                    fullFalse = resultFromClassify[1]

        elif nounResultFromSearchTrie[2]: #True if superword, else suffix
            ##print("Superword from noun trie: ",nounResultFromSearchTrie)
            resultFromClassify = classifySuperword(nounResultFromSearchTrie[0], noun_train, testWord, testDict[key])
            if resultFromClassify[0]:
                found = True
                ##print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            else:
                if len(resultFromClassify)==3:
                    halfFalse = resultFromClassify[1] + resultFromClassify[2]
                elif len(resultFromClassify)==2:
                    fullFalse = resultFromClassify[1]
        else:
            ##print("Suffix from noun trie: ",nounResultFromSearchTrie)
            resultFromClassify = classifySuffix(nounResultFromSearchTrie[0], noun_train, testWord, testDict[key])
            if resultFromClassify[0]:
                ##print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                found = True
            else:
                if len(resultFromClassify)==3:
                    halfFalse = resultFromClassify[1] + resultFromClassify[2]
                elif len(resultFromClassify)==2:
                    fullFalse = resultFromClassify[1]

        if adjResultFromSearchTrie[1] and not found:
            ##print("Subword from adjective trie: ",adjResultFromSearchTrie)
            resultFromClassify = classifySubword(adjResultFromSearchTrie[0], adjective_train, testWord, testDict[key])
            if resultFromClassify[0]:
                found = True
                ##print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            else:
                if len(resultFromClassify)==3:
                    halfFalse = resultFromClassify[1] + resultFromClassify[2]
                elif len(resultFromClassify)==2:
                    fullFalse = resultFromClassify[1]

        elif adjResultFromSearchTrie[2] and not found:
            ##print("Superword from adjective trie: ",adjResultFromSearchTrie)
            resultFromClassify = classifySuperword(adjResultFromSearchTrie[0], adjective_train, testWord, testDict[key])
            if resultFromClassify[0]:
                found = True
                ##print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            else:
                if len(resultFromClassify)==3:
                    halfFalse = resultFromClassify[1] + resultFromClassify[2]
                elif len(resultFromClassify)==2:
                    fullFalse = resultFromClassify[1]
        elif not found:
            ##print("Suffix from adjective trie: ",adjResultFromSearchTrie)
            resultFromClassify = classifySuffix(adjResultFromSearchTrie[0], adjective_train, testWord, testDict[key])
            if resultFromClassify[0]:
                ##print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                found = True
            else:
                if len(resultFromClassify)==3:
                    halfFalse = resultFromClassify[1] + resultFromClassify[2]
                elif len(resultFromClassify)==2:
                    fullFalse = resultFromClassify[1]


        if verbResultFromSearchTrie[1] and not found:
            ##print("Subword from verb trie: ",verbResultFromSearchTrie)
            resultFromClassify = classifySubword(verbResultFromSearchTrie[0], verb_train, testWord, testDict[key])
            if resultFromClassify[0]:
                found = True
                ##print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            else:
                if len(resultFromClassify)==3:
                    halfFalse = resultFromClassify[1] + resultFromClassify[2]
                elif len(resultFromClassify)==2:
                    fullFalse = resultFromClassify[1]

        elif verbResultFromSearchTrie[2] and not found:
            ##print("Superword from verb trie: ",verbResultFromSearchTrie)
            resultFromClassify = classifySuperword(verbResultFromSearchTrie[0], verb_train, testWord, testDict[key])
            if resultFromClassify[0]:
                ##print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                found = True
            else:
                if len(resultFromClassify)==3:
                    halfFalse = resultFromClassify[1] + resultFromClassify[2]
                elif len(resultFromClassify)==2:
                    fullFalse = resultFromClassify[1]

        elif not found:
            ##print("Suffix from verb trie: ",verbResultFromSearchTrie)
            resultFromClassify = classifySuffix(verbResultFromSearchTrie[0], verb_train, testWord, testDict[key])
            if resultFromClassify[0]:
                ##print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                found = True
            else:
                if len(resultFromClassify)==3:
                    halfFalse = resultFromClassify[1] + resultFromClassify[2]
                elif len(resultFromClassify)==2:
                    fullFalse = resultFromClassify[1]

        if pronResultFromSearchTrie[1] and not found:
            ##print("Subword from pronoun trie: ",pronResultFromSearchTrie)
            resultFromClassify = classifySubword(pronResultFromSearchTrie[0], pronoun_train, testWord, testDict[key])
            if resultFromClassify[0]:
                found = True
                ##print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            else:
                if len(resultFromClassify)==3:
                    halfFalse = resultFromClassify[1] + resultFromClassify[2]
                elif len(resultFromClassify)==2:
                    fullFalse = resultFromClassify[1]

        elif pronResultFromSearchTrie[2] and not found:
            ##print("Superword from pronoun trie: ",pronResultFromSearchTrie)
            resultFromClassify = classifySuperword(pronResultFromSearchTrie[0], pronoun_train, testWord, testDict[key])
            if resultFromClassify[0]:
                found = True
                ##print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            else:
                if len(resultFromClassify)==3:
                    halfFalse = resultFromClassify[1] + resultFromClassify[2]
                elif len(resultFromClassify)==2:
                    fullFalse = resultFromClassify[1]

        elif not found:
            ##print("Suffix from pronoun trie: ",pronResultFromSearchTrie)
            resultFromClassify = classifySuffix(pronResultFromSearchTrie[0], pronoun_train, testWord, testDict[key])
            if resultFromClassify[0]:
                ##print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                found = True
            else:
                if len(resultFromClassify)==3:
                    halfFalse = resultFromClassify[1] + resultFromClassify[2]
                elif len(resultFromClassify)==2:
                    fullFalse = resultFromClassify[1]

        if numResultFromSearchTrie[1] and not found:
            ##print("Subword from numeral trie: ",numResultFromSearchTrie)
            resultFromClassify = classifySubword(numResultFromSearchTrie[0], numeral_train, testWord, testDict[key])
            if resultFromClassify[0]:
                found = True
                ##print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            else:
                if len(resultFromClassify)==3:
                    halfFalse = resultFromClassify[1] + resultFromClassify[2]
                elif len(resultFromClassify)==2:
                    fullFalse = resultFromClassify[1]

        elif numResultFromSearchTrie[2] and not found:
            ##print("Superword from numeral trie: ",numResultFromSearchTrie)
            resultFromClassify = classifySuperword(numResultFromSearchTrie[0], numeral_train, testWord, testDict[key])
            if resultFromClassify[0]:
                ##print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                found = True
            else:
                if len(resultFromClassify)==3:
                    halfFalse = resultFromClassify[1] + resultFromClassify[2]
                elif len(resultFromClassify)==2:
                    fullFalse = resultFromClassify[1]

        elif not found:
            ##print("Suffix from numeral trie: ",numResultFromSearchTrie)
            resultFromClassify = classifySuffix(numResultFromSearchTrie[0], numeral_train, testWord, testDict[key])
            if resultFromClassify[0]:
                ##print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                found = True
            else:
                if len(resultFromClassify)==3:
                    halfFalse = resultFromClassify[1] + resultFromClassify[2]
                elif len(resultFromClassify)==2:
                    fullFalse = resultFromClassify[1]

        if found: pogodija+=1
        else:
            falija+=1
            if halfFalse:
                confusionMatrix(halfFalse)
            else:
                confusionMatrix(fullFalse)
        k="pogodija: "+str(pogodija)+ "\nfalija: "+str(falija)
        with open('guessOrfail.txt', 'w') as outfile:
            json.dump(k, outfile)




dot = Digraph()
dot.node('0', 'ROOT')
dot.format = 'svg'
trainsize = 0.95
separate(trainsize)
t=time.time()
decideFromTrainTries()
print(time.time()-t)
x='\t\tACTUAL\n\t  _____________________________\n\t\t N     V     A     P     M     ?\n'
x+='\tP|\n\tR| N     '+str(noun)+'     '+str(verbSaidNoun)+'     '+str(adjSaidNoun)+'     '+str(pronSaidNoun)+'     '+str(numSaidNoun)
x+='\n\tE|\n\tD| V     '+str(nounSaidVerb)+'     '+str(ver)+'     '+str(adjSaidVerb)+'     '+str(pronSaidVerb)+'     '+str(numSaidVerb)
x+='\n\tI| \n\tC| A     '+str(nounSaidAdj)+'     '+str(verbSaidAdj)+'     '+str(adj)+'     '+str(pronSaidAdj)+'     '+str(numSaidAdj)
x+='\n\tT|\n\tI| M     '+str(nounSaidNum)+'     '+str(verbSaidNum)+'     '+str(adjSaidNum)+'     '+str(pronSaidNum)+'     '+str(num)
x+='\n\tO|\n\tN| P     '+str(nounSaidPron)+'     '+str(verbSaidPron)+'     '+str(adjSaidPron)+'     '+str(pron)+'     '+str(numSaidPron)
x+='\n\t |\n\t | ?     '+str(nounSaidUnknown)+'     '+str(verbSaidUnknown)+'     '+str(adjSaidUnknown)+'     '+str(pronSaidUnknown)+'     '+str(numSaidUnknown)+'     '+str(unknown)
print(x)
#dot.render("..//trieDot.gv", view=True)