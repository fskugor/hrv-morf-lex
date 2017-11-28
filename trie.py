import os
import json
from pprint import pprint
from graphviz import Digraph
from hmldb import HmlDB
import random
import pickle
import time
import nltk

noun_, adj_, ver_, pron_, num_ = 0, 0, 0, 0, 0
nounSaidAdj, nounSaidVerb, nounSaidPron, nounSaidNum = 0, 0, 0, 0
adjSaidNoun, adjSaidVerb, adjSaidPron, adjSaidNum = 0, 0, 0, 0
verbSaidNoun, verbSaidPron, verbSaidAdj, verbSaidNum = 0, 0, 0, 0
pronSaidNoun, pronSaidAdj, pronSaidNum, pronSaidVerb = 0, 0, 0, 0
numSaidNoun, numSaidAdj, numSaidVerb, numSaidPron = 0, 0, 0, 0
nounMorphemes, adjMorphemes, verMorphemes, pronMorphemes, numMorphemes,  = 0, 0, 0, 0, 0
unknown = 0
failedadjMorphemes, failedverMorphemes, failedpronMorphemes, failednumMorphemes,failednounMorphemes = 0, 0, 0, 0, 0
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

def searchLongestSuffix(trie, word):
    current = trie[0]
    j = len(word)-1
    suffix=''
    currentNoOfWordsPassThrough=0
    while(word[j] in current):
        suffix+=word[j]
        currentNoOfWordsPassThrough = current.get(word[j])[1]
        current = current.get(word[j])[0]
        if j==1: break
        j-=1
    return [suffix[::-1],currentNoOfWordsPassThrough]# returns either suffix or subword(if exists)
def printify(trie):
    for k in trie[0]: #recursive function for adding nodes and connect them to edges for visualization to .dot file
        if(((trie[0])[k])[4] == True):
            dot.edge('0',str(((trie[0])[k])[3])) #connect node to root
        dot.node(str(((trie[0])[k])[3]), k+' |'+str(((trie[0])[k])[1])+'| '+str(((trie[0])[k])[2]))#describe node
        for j in ((trie[0])[k])[0]:
            dot.edge(str(((trie[0])[k])[3]),str(((((trie[0])[k])[0])[j])[3])) #connect node with next node
        printify((trie[0])[k])

def separate(trainsize):
    print("Enter separate()")
    db = HmlDB("../hml.db")
    seed, triple, br, noun_train, test = 125, {}, 0, {}, {}
    adjective_train, verb_train, pronoun_train, numeral_train = {}, {}, {}, {}
    adjective, verb, noun, pronoun, numeral = {}, {}, {}, {}, {}
    nounListForMaxent, adjectiveListForMaxent, verbListForMaxent, pronounListForMaxent, numeralListForMaxent = [], [], [], [], []
    allItems = HmlDB.select_all(db) # get all triples from db
    random.seed(seed)
    random.shuffle(allItems)
    nouns_count=HmlDB.count_tokens_by_msd(db,"N%")
    adj_count=HmlDB.count_tokens_by_msd(db,"A%")
    verb_count=HmlDB.count_tokens_by_msd(db,"V%")
    pron_count=HmlDB.count_tokens_by_msd(db,"P%")
    num_count=HmlDB.count_tokens_by_msd(db,"M%")
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
        if br>(nouns_count*trainsize):
            for j in noun[key]:
                if not key in test: test[key] = [(j[0],j[1])]
                else: test[key] += [(j[0],j[1])]
        else:
            for j in noun[key]:
                br+=1
                if not key in noun_train: noun_train[key] = [(j[0],j[1])]
                else: noun_train[key] += [(j[0],j[1])]
                nounListForMaxent+=[(j[0],j[1][0])]

    br = 0
    for key in adjective:
        if br>(adj_count*trainsize):
            for j in adjective[key]:
                if not key in test: test[key] = [(j[0],j[1])]
                else: test[key] += [(j[0],j[1])]
        else:
            for j in adjective[key]:
                br+=1
                if not key in adjective_train: adjective_train[key] = [(j[0],j[1])]
                else: adjective_train[key] += [(j[0],j[1])]
                adjectiveListForMaxent+=[(j[0],j[1][0])]

    br = 0
    for key in verb:
        if br>(verb_count*trainsize):
            for j in verb[key]:
                if not key in test: test[key] = [(j[0],j[1])]
                else: test[key] += [(j[0],j[1])]
        else:
            for j in verb[key]:
                br+=1
                if not key in verb_train: verb_train[key] = [(j[0],j[1])]
                else: verb_train[key] += [(j[0],j[1])]
                verbListForMaxent+=[(j[0],j[1][0])]

    br = 0
    for key in pronoun:
        if br>(pron_count*trainsize):
            for j in pronoun[key]:
                if not key in test: test[key] = [(j[0],j[1])]
                else: test[key] += [(j[0],j[1])]
        else:
            for j in pronoun[key]:
                br+=1
                if not key in pronoun_train: pronoun_train[key] = [(j[0],j[1])]
                else: pronoun_train[key] += [(j[0],j[1])]
                pronounListForMaxent+=[(j[0],j[1][0])]

    br = 0
    for key in numeral:
        if br>(num_count*trainsize):
            for j in numeral[key]:
                if not key in test: test[key] = [(j[0],j[1])]
                else: test[key] += [(j[0],j[1])]
        else:
            for j in numeral[key]:
                br+=1
                if not key in numeral_train: numeral_train[key] = [(j[0],j[1])]
                else: numeral_train[key] += [(j[0],j[1])]
                numeralListForMaxent+=[(j[0],j[1][0])]

    write('../test.pickle', test)  # write all to files,
    write('../trainNounTrie.pickle', put([{}], noun_train))
    write('../trainAdjectiveTrie.pickle', put([{}], adjective_train))
    write('../trainVerbTrie.pickle', put([{}], verb_train))
    write('../trainPronounTrie.pickle', put([{}], pronoun_train))
    write('../trainNumeralTrie.pickle', put([{}], numeral_train))
    write('../trainNounDictionary.pickle', noun_train)
    write('../trainAdjectiveDictionary.pickle', adjective_train)
    write('../trainVerbDictionary.pickle', verb_train)
    write('../trainPronounDictionary.pickle', pronoun_train)
    write('../trainNumeralDictionary.pickle', numeral_train)
    write('../numeralListForMaxent.pickle', numeralListForMaxent)
    write('../pronounListForMaxent.pickle', pronounListForMaxent)
    write('../verbListForMaxent.pickle', verbListForMaxent)
    write('../adjectiveListForMaxent.pickle', adjectiveListForMaxent)
    write('../nounListForMaxent.pickle', nounListForMaxent)
    print("Leaving separate()")

def confusionMatrix(fail):
    global nounSaidAdj, nounSaidVerb, nounSaidPron, nounSaidNum
    global adjSaidNoun, adjSaidVerb, adjSaidPron, adjSaidNum
    global verbSaidNoun, verbSaidPron, verbSaidAdj, verbSaidNum
    global pronSaidNoun, pronSaidAdj, pronSaidNum, pronSaidVerb
    global numSaidNoun, numSaidAdj, numSaidVerb, numSaidPron
#fail is string, for example NA, N is actual type of test word, A is prediction
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

def compare(original, generated, testWord):
    originalGroupByToken, generatedGroupByToken, br, guessedMsd, generatedKeys, originalKeys, msdFoundInGeneratedModel = {}, {}, 0, 0, [], [], 0
    global nounMorphemes, adjMorphemes, verMorphemes, pronMorphemes, numMorphemes
    global failedadjMorphemes, failedverMorphemes, failedpronMorphemes, failednumMorphemes,failednounMorphemes
    for pair in original:
        if not pair[0] in originalGroupByToken: originalGroupByToken[pair[0]]=[pair[1]]
        else: originalGroupByToken[pair[0]]+=[pair[1]]
    for pair in generated:
        if not pair[0] in generatedGroupByToken: generatedGroupByToken[pair[0]]=[pair[1]]
        else: generatedGroupByToken[pair[0]]+=[pair[1]]
    generatedKeys = [key for key in generatedGroupByToken]
    originalKeys = [key for key in originalGroupByToken]
    if set(generatedKeys) == set(originalKeys) and original[0][1][0]==generated[0][1][0]:
        for key in originalGroupByToken:
            for msd in originalGroupByToken[key]:
                if msd in generatedGroupByToken[key]:
                    br+=1
                    break
        guessedMsd=br/len(originalGroupByToken)
        if br == len(originalGroupByToken):
            guessedMsd = 1
        elif br == 0: guessedMsd = 0
        f = open('msd.txt', 'a')
        f.write(str(guessedMsd)+', '+original[0][1][0]+'\n')
        f = open('GuessedGenerating.txt', 'a')
        f.write(str(generated)+' <-----GENERATED|'+testWord+'|ORIGINAL------> '+str(original)+'\n\n')
        return True
    elif set(generatedKeys) != set(originalKeys) and original[0][1][0]==generated[0][1][0]: #both classificator
        if generated[0][1][0] == 'N': failednounMorphemes+=1
        elif generated[0][1][0] == 'A': failedadjMorphemes+=1
        elif generated[0][1][0] == 'V': failedverMorphemes+=1
        elif generated[0][1][0] == 'P': failedpronMorphemes+=1
        elif generated[0][1][0] == 'M': failednumMorphemes+=1
        f = open('wrongGeneratedMorphemes.txt', 'a')
        f.write(str(generated)+' <-----GENERATED||ORIGINAL------> '+str(original)+'\n\n')
    return False

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
        generatedTestModel, testModelOrigin = [], []
        pairs = pairs_train(lemma)
        subword = len(resultFromSearchTrie)
        preffixTest = testWord[:-subword]
        for i in pairs: #zamijenimo svaki prefix od treninga sa našim od testa
            generatedTestModel += [(preffixTest.lower()+i[0],i[1])]
        for j in testPairs:
            testModelOrigin +=[(j[0].lower(),j[1])]
        result = compare(testModelOrigin,generatedTestModel, resultFromSearchTrie)
        return result
    else: return False

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
        generatedTestModel, testModelOrigin = [], []
        pairs = pairs_train(lemma)
        test=len(testWord)
        preffixTrain = resultFromSearchTrie[:-test]
        #print(preffxTrain)
        for i in pairs: #zamijenimo svaki prefix od treninga sa našim od testa
            generatedTestModel += [(i[0][len(preffixTrain):].lower(),i[1])]
        for j in testPairs:
            testModelOrigin +=[(j[0].lower(),j[1])]
        result = compare(testModelOrigin,generatedTestModel, resultFromSearchTrie)
        return result
    else: return False

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
        lemmaEqSuffix = False
        generatedTestModel, testModelOrigin = [], []
        pairs = pairs_train(lemma)
        suffix = len(resultFromSearchTrie)
        if token == resultFromSearchTrie:
            preffixTrain = token[-suffix:]
            lemmaEqSuffix = True
        else: preffixTrain = token[:-suffix]
        preffixTest = testWord[:-suffix]
        for i in pairs: #zamijenimo svaki prefix od treninga sa našim od testa
            if lemmaEqSuffix:
                generatedTestModel += [(preffixTest.lower()+i[0],i[1])]
            else:
                generatedTestModel += [(preffixTest.lower()+i[0][len(preffixTrain):].lower(),i[1])]
        for j in testPairs:
            testModelOrigin +=[(j[0].lower(),j[1])]
        result = compare(testModelOrigin,generatedTestModel, resultFromSearchTrie)
        return result
    else: return False

def generateForms(trainTrie, testWord, _train, testDict):
    resultFromSearchTrie = search(trainTrie, testWord)
    found = False
    if resultFromSearchTrie[1]:
        resultFromClassify = classifySubword(resultFromSearchTrie[0], _train, testWord, testDict)
        if resultFromClassify:
            found = True
    elif resultFromSearchTrie[2] and not found:
        resultFromClassify = classifySuperword(resultFromSearchTrie[0], _train, testWord, testDict)
        if resultFromClassify:
            found = True
    elif not found:
        resultFromClassify = classifySuffix(resultFromSearchTrie[0], _train, testWord, testDict)
        if resultFromClassify:
            found = True
    return found

def suffixTrieClassify():
    global unknown
    global noun_, adj_, ver_, pron_, num_
    nounTrainTrie = read('../trainNounTrie.pickle') #nested dictionaries -->tries
    adjectiveTrainTrie = read('../trainAdjectiveTrie.pickle')
    verbTrainTrie = read('../trainVerbTrie.pickle')
    pronounTrainTrie = read('../trainPronounTrie.pickle')
    numeralTrainTrie = read('../trainNumeralTrie.pickle')
    noun_train = read('../trainNounDictionary.pickle') #dictionaries
    adjective_train = read('../trainAdjectiveDictionary.pickle')
    verb_train = read('../trainVerbDictionary.pickle')
    pronoun_train = read('../trainPronounDictionary.pickle')
    numeral_train = read('../trainNumeralDictionary.pickle')
    pogodija, falija = 0, 0
    testDict = read('../test.pickle') #dict
    lenTest=len(testDict)
    print("Test length: ",lenTest)
    for key in testDict: # prolazimo kroz svaki prikaz i uzmemo oblik(token=lemma) za testiranje
        words=set()
        msd = testDict[key][0][1][0]
        for pair in testDict[key]:
            words.add(pair[0])
        # random.shuffle(words)
        # testWord=words[0]
        for testWord in words:
            nounResultFromSearchTrie, adjResultFromSearchTrie, pronResultFromSearchTrie = '', '', ''
            verbResultFromSearchTrie, numResultFromSearchTrie = '', ''
            nounTrieLongestSuffix = searchLongestSuffix(nounTrainTrie, testWord)
            verbTrieLongestSuffix = searchLongestSuffix(verbTrainTrie, testWord)
            adjTrieLongestSuffix = searchLongestSuffix(adjectiveTrainTrie, testWord)
            pronTrieLongestSuffix = searchLongestSuffix(pronounTrainTrie, testWord)
            numTrieLongestSuffix = searchLongestSuffix(numeralTrainTrie, testWord)
            suffix = [len(nounTrieLongestSuffix[0]), len(verbTrieLongestSuffix[0]), \
                      len(adjTrieLongestSuffix[0]), len(pronTrieLongestSuffix[0]), len(numTrieLongestSuffix[0])]
            suffixMax = max(suffix)
            trieFromWhichClassifyStarts = [] #array of possible type of test word, if the word has more max suffixes
            if suffix[0]==suffixMax: trieFromWhichClassifyStarts+=[('N', nounTrieLongestSuffix[1])]
            if suffix[1]==suffixMax: trieFromWhichClassifyStarts+=[('V', verbTrieLongestSuffix[1])]
            if suffix[2]==suffixMax: trieFromWhichClassifyStarts+=[('A', adjTrieLongestSuffix[1])]
            if suffix[3]==suffixMax: trieFromWhichClassifyStarts+=[('P', pronTrieLongestSuffix[1])]
            if suffix[4]==suffixMax: trieFromWhichClassifyStarts+=[('M', numTrieLongestSuffix[1])]
            found = False
            pair=[]
            if len(trieFromWhichClassifyStarts)>1:
                for i in range(len(trieFromWhichClassifyStarts)-1):
                    for j in range(i+1, len(trieFromWhichClassifyStarts)):
                        if trieFromWhichClassifyStarts[i][1]>trieFromWhichClassifyStarts[j][1]:
                            pair=trieFromWhichClassifyStarts[i]
                        else: pair=trieFromWhichClassifyStarts[j]
            else: pair=trieFromWhichClassifyStarts[0]
            if pair[0]=='N' and msd == 'N':
                noun_+=1
                pogodija+=1
                found = generateForms(nounTrainTrie, testWord, noun_train, testDict[key])
            elif pair[0]=='V' and msd == 'V':
                ver_+=1
                pogodija+=1
                found = generateForms(verbTrainTrie, testWord, verb_train, testDict[key])
            elif pair[0]=='A' and msd =='A':
                adj_+=1
                pogodija+=1
                found = generateForms(adjectiveTrainTrie, testWord, adjective_train, testDict[key])
            elif pair[0]=='P' and msd == 'P':
                pron_+=1
                pogodija+=1
                found = generateForms(pronounTrainTrie, testWord, pronoun_train, testDict[key])
            elif pair[0]=='M' and msd =='M':
                num_+=1
                pogodija+=1
                found = generateForms(numeralTrainTrie, testWord, numeral_train, testDict[key])
            else:
                confusionMatrix(msd+pair[0])
                falija+=1
            if found:
                unknown+=1
                break


        k='pogodija tip riječi: '+str(pogodija)+ '\n falija tip riječi: '+str(falija)+'\n'+ 'Točno izgenerirani:'+str(unknown)
        with open('guessOrfail.txt', 'w') as outfile:
            json.dump(k, outfile)


def featuresForMaxent(word):
    if len(word)>10:return{'First and last letter':word[-6:]}
    if len(word)==10:return{'First and last letter':word[-5:]}
    if len(word)==5:return{'First and last letter':word[-3:]}
    if len(word)==4:return{'First and last letter':word[0]+word[-1]}
    if len(word)>5 and len(word)<10: return {'Last 3 letters':word[-4:]}
    if len(word)<=3: return {'Last 2 letters':word[-2:]}


def maxentClassify():
    pogodija, failedType = 0, 0
    wholeTrain = set()
    random.seed(4)
    global unknown
    global noun_, adj_, ver_, pron_, num_
    nounTrainTrie = read('../trainNounTrie.pickle') #nested dictionaries -->tries
    adjectiveTrainTrie = read('../trainAdjectiveTrie.pickle')
    verbTrainTrie = read('../trainVerbTrie.pickle')
    pronounTrainTrie = read('../trainPronounTrie.pickle')
    numeralTrainTrie = read('../trainNumeralTrie.pickle')
    noun_train = read('../trainNounDictionary.pickle') #dictionaries
    adjective_train = read('../trainAdjectiveDictionary.pickle')
    verb_train = read('../trainVerbDictionary.pickle')
    pronoun_train = read('../trainPronounDictionary.pickle')
    numeral_train = read('../trainNumeralDictionary.pickle')
    numeral = read('../numeralListForMaxent.pickle')
    pronoun = read('../pronounListForMaxent.pickle')
    verb = read('../verbListForMaxent.pickle')
    adjective = read('../adjectiveListForMaxent.pickle')
    noun = read('../nounListForMaxent.pickle')
    testDict = read('../test.pickle')#dict
    wholeTrain = noun + adjective + verb + pronoun + numeral
    random.shuffle(wholeTrain)
    trainingSet=[(featuresForMaxent(word), typeOf) for(word, typeOf) in wholeTrain]
    testLen=len(testDict)
    print("Train lenght: ", len(trainingSet))
    print("Test lenght: ", testLen)
    #devTestSet= len(trainingSet)*0.3
    print("start train")
    classifierMaxE=nltk.MaxentClassifier.train(trainingSet)
    print("finished")
    for key in testDict: # prolazimo kroz svaki prikaz i uzmemo oblik(token=lemma) za testiranje
        msd = testDict[key][0][1][0]
        for pair in testDict[key]:
            testWord=pair[0]
            found  = False
            msdFromClassifier = classifierMaxE.classify(featuresForMaxent(testWord))
            if msd=='N' and msdFromClassifier==msd:
                noun_+=1
                pogodija+=1
                found = generateForms(nounTrainTrie, testWord, noun_train, testDict[key])
            elif msd=='V' and msdFromClassifier==msd:
                ver_+=1
                pogodija+=1
                found = generateForms(verbTrainTrie, testWord, verb_train, testDict[key])
            elif msd=='A' and msdFromClassifier==msd:
                adj_+=1
                pogodija+=1
                found = generateForms(adjectiveTrainTrie, testWord, adjective_train, testDict[key])
            elif msd=='M' and msdFromClassifier==msd:
                num_+=1
                pogodija+=1
                found = generateForms(numeralTrainTrie, testWord, numeral_train, testDict[key])
            elif msd=='P' and msdFromClassifier==msd:
                pron_+=1
                pogodija+=1
                found = generateForms(pronounTrainTrie, testWord, pronoun_train, testDict[key])
            else:
                failedType+=1
                confusionMatrix(msd + msdFromClassifier)
            if msdFromClassifier!='N' and msdFromClassifier !='A' and msdFromClassifier !='P' and msdFromClassifier !='M' and msdFromClassifier !='V':
                unknown+=1
                print(testWord, msdFromClassifier)
            if found: break

        k='pogodija tip: '+str(pogodija)+ '\nfalija klasificirat tip: '+str(failedType)+'\nOOV: '+str(unknown)
        with open('guessOrfailMaxEnt.txt', 'w') as outfile:
            json.dump(k, outfile)

dot = Digraph()
dot.node('0', 'ROOT')
dot.format = 'svg'
trainsize = 0.8
separate(trainsize)
# t=time.time()
# suffixTrieClassify()
# print(time.time()-t)
k=time.time()
print("MAXENT")
maxentClassify()
print(time.time()-k)

x='\t\tACTUAL\n\t  _____________________________\n\t\t N        V        A        P        M\n'
x+='\tP|\n\tR| N     '+str(noun_)+'        '+str(verbSaidNoun)+'        '+str(adjSaidNoun)+'        '+str(pronSaidNoun)+'        '+str(numSaidNoun)
x+='\n\tE|\n\tD| V     '+str(nounSaidVerb)+'        '+str(ver_)+'        '+str(adjSaidVerb)+'        '+str(pronSaidVerb)+'        '+str(numSaidVerb)
x+='\n\tI| \n\tC| A     '+str(nounSaidAdj)+'        '+str(verbSaidAdj)+'        '+str(adj_)+'        '+str(pronSaidAdj)+'        '+str(numSaidAdj)
x+='\n\tT|\n\tI| M     '+str(nounSaidNum)+'        '+str(verbSaidNum)+'        '+str(adjSaidNum)+'        '+str(pronSaidNum)+'        '+str(num_)
x+='\n\tO|\n\tN| P     '+str(nounSaidPron)+'        '+str(verbSaidPron)+'        '+str(adjSaidPron)+'        '+str(pron_)+'        '+str(numSaidPron)
x+='\n\t |\n\t | ?     '+' --->Unknown words'+str(unknown)+'\nFailed morphemes: N,V,A,M,P -'+str(failednounMorphemes)+', '+str(failedverMorphemes)+', '+str(failedadjMorphemes)+', '+str(failednumMorphemes)+', '+str(failedpronMorphemes)
print(x)
#dot.render("..//trieDot.gv", view=True)