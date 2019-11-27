class Node(object):
    def __init__(self, word, index):
        self.word = word
        self.index = index
        self.probabilities = {'V': 0, 'N': 0, 'ART': 0, 'P': 0}
        self.normalizedProbabilities = {}

    def normalizeProbabilities(self):
        total = sum(self.probabilities.values())
        self.normalizedProbabilities = {}
        for p in self.probabilities:
            self.normalizedProbabilities[p] = self.probabilities[p] / total
        return "'" + self.word +"'" +  ' normalized probabilities:' + str(self.normalizedProbabilities)

    def __repr__(self):
        return f"*{self.word}*: \nIndex: {self.index}\nNormalized Probabilities: {self.normalizedProbabilities}\nRaw Probabilities: {self.probabilities}\n\n"

