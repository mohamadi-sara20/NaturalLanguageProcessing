'''
    POS tagging algorithms; ie Recursive Forward, Forward and Backward Algorithms. (HW#2, HW#3, HW#4)
    File name: SimpleProbabilisticParser.py
    Author: Sara Shahmohamadi
    Course: Natural Language Processing Fall 2019
    Date last modified: 10/22/2019
    Python Version: 3.6
'''

import pandas as pd

from Node import Node


class SimpleProbabilisticTagger(object):

    def __init__(self, lexicalGenerationFile, transitionFile):
        self.lexicalGenerations = SimpleProbabilisticTagger.readProbabilities(lexicalGenerationFile)
        self.transitions = self.readProbabilities(transitionFile)

    @staticmethod
    def extractLexicalGenerationProbabilities(filename, outfilename):
        df = pd.read_csv(filename)
        with open(outfilename, 'w') as f:
            f.write('word,tag,probability\n')
            for i in range(1, len(df.columns) - 1):
                for j in range(len(df[df.columns[i]]) - 1):
                    denominator = df[df.columns[i]][df.index[-1]]
                    lexicalGen = df[df.columns[i]][j] / denominator
                    f.write(df['WORD'][j] + ',' + df.columns[i] + ',' + str(lexicalGen) + '\n')

    @staticmethod
    def readProbabilities(filename):
        df = pd.read_csv(filename)
        feature_series = df.transpose()
        return feature_series

    def getLexicalProbability(self, word):
        possibleTags = {}
        for i in range(len(self.lexicalGenerations.columns)):
            if self.lexicalGenerations[i]['word'] == word:
                possibleTags[self.lexicalGenerations[i]['tag']] = self.lexicalGenerations[i]['probability']
        return possibleTags

    def getTransitionProbabilities(self, pos1, pos2):
        for i in range(len(self.transitions.columns)):
            if self.transitions[i]['tag1'] == pos1 and self.transitions[i]['tag2'] == pos2:
                return self.transitions[i]['probability']
        return 0.0001

    # HW2 (RECURSIVE FORWARD IMPLEMENTED) PART#1
    def enumerationPredictWord(self, i, words):
        probabilities = {'V': 0, 'N': 0, 'ART': 0, 'P': 0}
        if i == 0:
            currentTags = self.getLexicalProbability(words[i])
            for tag in currentTags:
                lexicalGenerationProbability = currentTags[tag]
                probabilities[tag] = self.getTransitionProbabilities(tag, '^') * lexicalGenerationProbability
            return probabilities

        currentTags = self.getLexicalProbability(words[i])
        for tag in currentTags:
            lexicalGenerationProbability = currentTags[tag]
            for former_tag in ['V', 'P', 'ART', 'N']:
                probabilities[tag] += self.enumerationPredictWord(words=words, i=i - 1)[former_tag] \
                                      * lexicalGenerationProbability * \
                                      self.getTransitionProbabilities(tag, former_tag)
        return probabilities

    # HW2 (RECURSIVE FORWARD IMPLEMENTED) PART#2
    def enumerationPredictSentence(self, sentence):
        wordsInfo = []
        words = sentence.split()
        for i in range(len(words)):
            node = Node(word=words[i], index=i)
            wordsInfo.append(node)
        for i in range(len(words)):
            wordsInfo[i].probabilities = self.enumerationPredictWord(words=words, i=i)
            wordsInfo[i].normalizeProbabilities()
        return wordsInfo

    @staticmethod
    def showResult(wordsInfo):
        for i in wordsInfo:
            print(i)

    # HW2 RECURSIVE FORWARD ALGORITHM DISPLAY FUNCTION. WORK WITH THIS TO SEE AND CHECK THE RESULTS.
    def processWithEnumerationPredictPos(self, sentence):
        print()
        print('############################################################')
        print('ENUMERATION PROBABILISTIC RESULTS(RECURSIVE):')
        wordsInfo = self.enumerationPredictSentence(sentence)
        self.showResult(wordsInfo)
        print('############################################################')

    # HW3 (FORWARD ALGORITHM IMPLEMENTED)
    def forwardPredictPosTags(self, sentence):
        wordsInfo = []
        words = sentence.split()

        for i in range(len(words)):
            node = Node(word=words[i], index=i)
            wordsInfo.append(node)

        currentTags = self.getLexicalProbability(words[0])

        for tag in currentTags:
            lexicalGenerationProbability = currentTags[tag]
            wordsInfo[0].probabilities[tag] = self.getTransitionProbabilities(tag, '^') * lexicalGenerationProbability
        wordsInfo[0].normalizeProbabilities()

        for i in range(1, len(words)):
            currentTags = self.getLexicalProbability(words[i])
            for tag in currentTags:
                lexicalGenerationProbability = currentTags[tag]
                for former_tag in wordsInfo[i - 1].probabilities:
                    wordsInfo[i].probabilities[tag] += wordsInfo[i - 1].probabilities[former_tag] \
                                                       * lexicalGenerationProbability * \
                                                       self.getTransitionProbabilities(tag, former_tag)
            wordsInfo[i].normalizeProbabilities()
        return wordsInfo

    # HW3 FORWARD ALGORITHM DISPLAY FUNCTION. WORK WITH THIS TO SEE AND CHECK THE RESULTS.
    def processSentenceForwardPredictPosTags(self, sentence):
        print()
        print('############################################################')
        print('FORWARD ALGORITHM RESULTS:')
        wordsInfo = self.forwardPredictPosTags(sentence)
        self.showResult(wordsInfo)
        print('############################################################')

    # HW4 BACKWARD ALGORITHM IMPLEMENTED
    def backwardPredictPos(self, sentence):
        wordsInfo = []
        words = sentence.split()

        for i in range(len(words)):
            node = Node(word=words[i], index=i)
            wordsInfo.append(node)

        currentInd = len(words) - 1
        currentTags = self.getLexicalProbability(words[currentInd])

        #Initialization
        for currTag in currentTags:
            lexicalGenerationProbability = currentTags[currTag]
            # P(O|λ) * βT (i)
            # P(O|λ) = SUM(πi.bi(o1).β1(i)). We take βT (i) = 1.
            wordsInfo[currentInd].probabilities[currTag] = 1 * lexicalGenerationProbability
        wordsInfo[len(words) - 1].normalizeProbabilities()

        #Filling trellis
        for i in range(currentInd - 1, -1, -1):
            currentTags = self.getLexicalProbability(words[i])
            nextTags = self.getLexicalProbability(words[i + 1])
            for currTag in currentTags:
                for nextTag in wordsInfo[i + 1].probabilities:
                    lexicalGenerationProbability = nextTags[nextTag]
                    wordsInfo[i].probabilities[currTag] += lexicalGenerationProbability * \
                                                           self.getTransitionProbabilities(nextTag, currTag) * \
                                                           wordsInfo[i + 1].probabilities[nextTag]
            wordsInfo[i].normalizeProbabilities()

        return wordsInfo

    # HW4 BACKWARD ALGORITHM DISPLAY FUNCTION. WORK WITH THIS TO SEE AND CHECK THE RESULTS.
    def processWithBackwardPredictPos(self, sentence):
        print()
        print('############################################################')
        print('BACKWARD RESULTS:')
        wordsInfo = self.backwardPredictPos(sentence)
        self.showResult(wordsInfo)
        print('############################################################')


if __name__ == '__main__':
    SimpleProbabilisticTagger.extractLexicalGenerationProbabilities('frequency.csv',
                                                                    'lexicalGenerationProbabilities.csv')
    tagger = SimpleProbabilisticTagger('lexicalGenerationProbabilities.csv', 'transition.csv')
    # Recursive Forward aka HW#2 results:
    #tagger.processWithEnumerationPredictPos('the flies like flowers')
    # Forward algorithm HW#3 results:
    #tagger.processSentenceForwardPredictPosTags('a birds flies')
    # Backward algorithm HW#4 results:
    tagger.processWithBackwardPredictPos('a birds flies')
