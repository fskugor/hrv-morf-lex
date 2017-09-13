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
nounSaidUnknown, pronSaidUnknown, adjSaidUnknown, verbSaidUnknown, numSaidUnknown = 0, 0, 0, 0, 0

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
        lenght=len(word)
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
    print("Enter separate()")
    db = HmlDB("../hml.db")
    seed, triple, br, noun_train, test = 256, {}, 0, {}, {}
    adjective_train, verb_train, pronoun_train, numeral_train = {}, {}, {}, {}
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
            for j in triple[lemma]:
                if j[1][0] == 'N' or j[1][0] == 'A' or j[1][0] == 'V' or [1][0] == 'P' or j[1][0] == 'M':
                    if not lemma in test: test[lemma] = [(j[0],j[1])]
                    else: test[lemma] += [(j[0],j[1])]
        else:
            for j in triple[lemma]:
                if j[1][0] == 'N':
                    if not lemma in noun_train: noun_train[lemma] = [(j[0],j[1])]
                    else: noun_train[lemma] += [(j[0],j[1])]
                if j[1][0] == 'A':
                    if not lemma in adjective_train: adjective_train[lemma] = [(j[0],j[1])]
                    else: adjective_train[lemma] += [(j[0],j[1])]
                if j[1][0] == 'V':
                    if not lemma in verb_train: verb_train[lemma] = [(j[0],j[1])]
                    else: verb_train[lemma] += [(j[0],j[1])]
                if j[1][0] == 'P':
                    if not lemma in pronoun_train: pronoun_train[lemma] = [(j[0],j[1])]
                    else: pronoun_train[lemma] += [(j[0],j[1])]
                if j[1][0] == 'M':
                    if not lemma in numeral_train: numeral_train[lemma] = [(j[0],j[1])]
                    else: numeral_train[lemma] += [(j[0],j[1])]

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
    print("Leaving separate()")

def compare(original, generated):
    originalGroupByToken, generatedGroupByToken, br, guessedMsd, generatedKeys, originalKeys = {}, {}, 0, 0, [], []
    global noun, adj, ver, pron, num
    global nounSaidAdj, nounSaidVerb, nounSaidPron, nounSaidNum
    global adjSaidNoun, adjSaidVerb, adjSaidPron, adjSaidNum
    global verbSaidNoun, verbSaidPron, verbSaidAdj, verbSaidNum
    global pronSaidNoun, pronSaidAdj, pronSaidNum, pronSaidVerb
    global numSaidNoun, numSaidAdj, numSaidVerb, numSaidPron
    global nounSaidUnknown, pronSaidUnknown, adjSaidUnknown, verbSaidUnknown, numSaidUnknown
    for pair in original:
        if not pair[0] in originalGroupByToken: originalGroupByToken[pair[0]]=[pair[1]]
        else: originalGroupByToken[pair[0]]+=[pair[1]]
    for pair in generated:
        if not pair[0] in generatedGroupByToken: generatedGroupByToken[pair[0]]=[pair[1]]
        else: generatedGroupByToken[pair[0]]+=[pair[1]]
    generatedKeys = [key for key in generatedGroupByToken]
    originalKeys = [key for key in originalGroupByToken]
    if set(generatedKeys) == set(originalKeys) and original[0][1][0]==generated[0][1][0]:
        print(generatedGroupByToken)
        print('++++++++++++++++++++++++++++++++++++++++++++')
        print(originalGroupByToken)
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
        print("MSD : ", br, len(originalGroupByToken))
        f = open('msd.txt', 'a')
        f.write(str(guessedMsd)+'\n')
        return True
    elif set(generatedKeys) == set(originalKeys) and original[0][1][0]!=generated[0][1][0]:
        if original[0][1][0] == 'N':
            if generated[0][1][0] == 'A': nounSaidAdj+=1
            elif generated[0][1][0] == 'V': nounSaidVerb+=1
            elif generated[0][1][0] == 'P': nounSaidPron+=1
            elif generated[0][1][0] == 'M': nounSaidNum+=1
        elif original[0][1][0] == 'A':
            if generated[0][1][0] == 'N': adjSaidNoun+=1
            elif generated[0][1][0] == 'V': adjSaidVerb+=1
            elif generated[0][1][0] == 'P': adjSaidPron+=1
            elif generated[0][1][0] == 'M': adjSaidNum+=1
        elif original[0][1][0] == 'V':
            if generated[0][1][0] == 'N': verbSaidNoun+=1
            elif generated[0][1][0] == 'A': verbSaidAdj+=1
            elif generated[0][1][0] == 'P': verbSaidPron+=1
            elif generated[0][1][0] == 'M': verbSaidNum+=1
        elif original[0][1][0] == 'P':
            if generated[0][1][0] == 'N': pronSaidNoun+=1
            elif generated[0][1][0] == 'A': pronSaidAdj+=1
            elif generated[0][1][0] == 'V': pronSaidVerb+=1
            elif generated[0][1][0] == 'M': pronSaidNum+=1
        elif original[0][1][0] == 'M':
            if generated[0][1][0] == 'N': numSaidNoun+=1
            elif generated[0][1][0] == 'A': numSaidAdj+=1
            elif generated[0][1][0] == 'V': numSaidVerb+=1
            elif generated[0][1][0] == 'P': numSaidPron+=1
        return False
    else:
        if original[0][1][0] == 'N': nounSaidUnknown+=1
        elif original[0][1][0] == 'A': adjSaidUnknown+=1
        elif original[0][1][0] == 'V': verbSaidUnknown+=1
        elif original[0][1][0] == 'P': pronSaidUnknown+=1
        elif original[0][1][0] == 'M': numSaidUnknown+=1
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

    guessed=False

    if pairs_train(lemma):
        print("Lemma za generiranje testnog prikaza: ",lemma)
        print("\nTrain token: ",resultFromSearchTrie)
        print("\nTestni token: ",testWord)
        generatedTestModel, testModelOrigin = [], []
        pairs = pairs_train(lemma)
        subword = len(resultFromSearchTrie)
        if(subword==len(testWord)): classifySuffix(resultFromSearchTrie, trainDict, testWord, testPairs)
        else:
            preffixTest = testWord[:-subword]
            for i in pairs: #zamijenimo svaki prefix od treninga sa našim od testa
                generatedTestModel += [(preffixTest.lower()+i[0],i[1])]
        for j in testPairs:
            testModelOrigin +=[(j[0].lower(),j[1])]
        if compare(testModelOrigin,generatedTestModel):
            guessed = True
            print("Test model found: ",generatedTestModel)
        else: print("Test model not found")
    else: return False
    return guessed

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

    guessed=False

    if pairs_train(lemma):
        print("Lemma za generiranje testnog prikaza: ",lemma)
        print("\nTrain token: ",lemma)
        print("\nTestni token: ",testWord)
        generatedTestModel, testModelOrigin = [], []
        pairs = pairs_train(lemma)
        test=len(testWord)
        preffixTrain = resultFromSearchTrie[:-test]
        print(preffixTrain)
        for i in pairs: #zamijenimo svaki prefix od treninga sa našim od testa
            generatedTestModel += [(i[0][len(preffixTrain):],i[1])]
        for j in testPairs:
            testModelOrigin +=[(j[0].lower(),j[1])]
        if compare(testModelOrigin,generatedTestModel):
            guessed = True
            print("Test model found: ",generatedTestModel)
        else: print("Test model not found")
    else: return False
    return guessed

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

    guessed=False

    if pairs_train(lemma):
        print("Lemma za generiranje testnog prikaza: ",lemma)
        print("\nTrain token: ",token)
        print("\nTestni token: ",testWord)
        lemmaEqSuffix = False
        generatedTestModel, testModelOrigin = [], []
        pairs = pairs_train(lemma)
        suffix = len(resultFromSearchTrie)
        if token == resultFromSearchTrie:
            preffixTrain = token[-suffix:]
            lemmaEqSuffix = True
        else: preffixTrain = token[:-suffix]
        preffixTest = testWord[:-suffix]
        print(preffixTrain)
        print(preffixTest)
        for i in pairs: #zamijenimo svaki prefix od treninga sa našim od testa
            if lemmaEqSuffix:
                generatedTestModel += [(preffixTest.lower()+i[0],i[1])]
            else:
                generatedTestModel += [(preffixTest.lower()+i[0][len(preffixTrain):].lower(),i[1])]
        for j in testPairs:
            testModelOrigin +=[(j[0].lower(),j[1])]
        if compare(testModelOrigin,generatedTestModel):
            guessed = True
            print("Test model found: ",generatedTestModel)
        else: print("Test model not found")
    else: return False
    return guessed

def decideFromTrainTries():
    db = HmlDB("..//hml.db")
    print("Enter decideFromTrainTries()")
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
    testDict = read('../test.pickle') #list
    for key in testDict: # prolazimo kroz svaki prikaz i uzmemo oblik(token=lemma) za testiranje
        # for j in testDict[key]:
        #     if j[0]==key:
        #         testWord=key
        #         break
        # print("Testna riječ: ",testWord)
        testWord=testDict[key][0][0] #oblik koji nije nužno jednak lemmi
        nounResultFromSearchTrie = search(nounTrainTrie, testWord)
        adjResultFromSearchTrie = search(adjectiveTrainTrie, testWord)
        verbResultFromSearchTrie = search(verbTrainTrie, testWord)
        pronResultFromSearchTrie = search(pronounTrainTrie, testWord)
        numResultFromSearchTrie = search(numeralTrainTrie, testWord)
        print(testWord)
        # print("noun trie: ",nounResultFromSearchTrie)
        # print("adjective trie: ",adjResultFromSearchTrie)
        # print("verb trie: ",verbResultFromSearchTrie)
        # print("pronoun trie: ",pronResultFromSearchTrie)
        # print("numeral trie: ",numResultFromSearchTrie)
        # print('\n----------------------------------------------------------------')
        found = False
        if nounResultFromSearchTrie[1]:
            print("Subword from noun trie: ",nounResultFromSearchTrie)
            if classifySubword(nounResultFromSearchTrie[0], noun_train, testWord, testDict[key]):
                found = True
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        elif nounResultFromSearchTrie[2]: #True if superword, else suffix
            print("Superword from noun trie: ",nounResultFromSearchTrie)
            if classifySuperword(nounResultFromSearchTrie[0], noun_train, testWord, testDict[key]):
                found = True
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        else:
            print("Suffix from noun trie: ",nounResultFromSearchTrie)
            if classifySuffix(nounResultFromSearchTrie[0], noun_train, testWord, testDict[key]):
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                found = True

        if adjResultFromSearchTrie[1] and not found:
            print("Subword from adjective trie: ",adjResultFromSearchTrie)
            if classifySubword(adjResultFromSearchTrie[0], adjective_train, testWord, testDict[key]):
                found = True
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        elif adjResultFromSearchTrie[2] and not found:
            print("Superword from adjective trie: ",adjResultFromSearchTrie)
            if classifySuperword(adjResultFromSearchTrie[0], adjective_train, testWord, testDict[key]):
                found = True
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        elif not found:
            print("Suffix from adjective trie: ",adjResultFromSearchTrie)
            if classifySuffix(adjResultFromSearchTrie[0], adjective_train, testWord, testDict[key]):
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                found = True


        if verbResultFromSearchTrie[1] and not found:
            print("Subword from verb trie: ",verbResultFromSearchTrie)
            if classifySubword(verbResultFromSearchTrie[0], verb_train, testWord, testDict[key]):
                found = True
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        elif verbResultFromSearchTrie[2] and not found:
            print("Superword from verb trie: ",verbResultFromSearchTrie)
            if classifySuperword(verbResultFromSearchTrie[0], verb_train, testWord, testDict[key]):
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                found = True
        elif not found:
            print("Suffix from verb trie: ",verbResultFromSearchTrie)
            if classifySuffix(verbResultFromSearchTrie[0], verb_train, testWord, testDict[key]):
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                found = True

        if pronResultFromSearchTrie[1] and not found:
            print("Subword from pronoun trie: ",pronResultFromSearchTrie)
            if classifySubword(pronResultFromSearchTrie[0], pronoun_train, testWord, testDict[key]):
                found = True
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        if pronResultFromSearchTrie[2] and not found:
            print("Superword from pronoun trie: ",pronResultFromSearchTrie)
            if classifySuperword(pronResultFromSearchTrie[0], pronoun_train, testWord, testDict[key]):
                found = True
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        elif not found:
            print("Suffix from pronoun trie: ",pronResultFromSearchTrie)
            if classifySuffix(pronResultFromSearchTrie[0], pronoun_train, testWord, testDict[key]):
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                found = True


        if numResultFromSearchTrie[1] and not found:
            print("Subword from numeral trie: ",numResultFromSearchTrie)
            if classifySubword(numResultFromSearchTrie[0], numeral_train, testWord, testDict[key]):
                found = True
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        if numResultFromSearchTrie[2] and not found:
            print("Superword from numeral trie: ",numResultFromSearchTrie)
            if classifySuperword(numResultFromSearchTrie[0], numeral_train, testWord, testDict[key]):
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                found = True
        elif not found:
            print("Suffix from numeral trie: ",numResultFromSearchTrie)
            if classifySuffix(numResultFromSearchTrie[0], numeral_train, testWord, testDict[key]):
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                found = True
        if found: pogodija+=1
        else:
            falija+=1
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
x='\t\tACTUAL\n\t  _____________________________\n\t\t N     V     A     P     M\n'
x+='\tP|\n\tR| N     '+str(noun)+'     '+str(verbSaidNoun)+'     '+str(adjSaidNoun)+'     '+str(pronSaidNoun)+'     '+str(numSaidNoun)
x+='\n\tE|\n\tD| V     '+str(nounSaidVerb)+'     '+str(ver)+'     '+str(adjSaidVerb)+'     '+str(pronSaidVerb)+'     '+str(numSaidVerb)
x+='\n\tI| \n\tC| A     '+str(nounSaidAdj)+'     '+str(verbSaidAdj)+'     '+str(adj)+'     '+str(pronSaidAdj)+'     '+str(numSaidAdj)
x+='\n\tT|\n\tI| M     '+str(nounSaidNum)+'     '+str(verbSaidNum)+'     '+str(adjSaidNum)+'     '+str(pronSaidNum)+'     '+str(num)
x+='\n\tO|\n\tN| P     '+str(nounSaidPron)+'     '+str(verbSaidPron)+'     '+str(adjSaidPron)+'     '+str(pron)+'     '+str(numSaidPron)
x+='\n\t |\n\t | ?     '+str(nounSaidUnknown)+'     '+str(verbSaidUnknown)+'     '+str(adjSaidUnknown)+'     '+str(pronSaidUnknown)+'     '+str(numSaidUnknown)
print(x)
#dot.render("..//trieDot.gv", view=True)