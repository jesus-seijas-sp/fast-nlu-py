import re
import unicodedata
import numpy as np

def normalize(str):
    return unicodedata.normalize('NFD', str).encode('ascii', 'ignore').decode('utf-8').lower()

def tokenize(text):
    return [x for x in re.split("[\\s,.!?;:([\\]'\"¡¿)/]+", text) if x]

def defaultProcessor(text):
    return tokenize(normalize(text))

class Encoder:
    def __init__(self, processor = None):
      self.processor = processor or defaultProcessor
      self.featureMap = {}
      self.numFeature = 0
      self.intentMap = {}
      self.intents = []   

    def learnIntent(self, intent):
        if intent not in self.intentMap:
            self.intentMap[intent] = len(self.intents)
            self.intents.append(intent)

    def learnFeature(self, feature):
        if feature not in self.featureMap:
            self.featureMap[feature] = self.numFeature
            self.numFeature += 1

    def encodeText(self, text, learn = False):
        dict = {}
        keys = []
        features = self.processor(text)
        for feature in features:
            if learn:
                self.learnFeature(feature)
            index = self.featureMap.get(feature)
            if index is not None and index not in dict:
                dict[index] = 1
                keys.append(index)
        return np.array(keys)

    def encode(self, text, intent, learn = False):
        if learn:
            self.learnIntent(intent)
        return [self.encodeText(text, learn), self.intentMap.get(intent)]
        #     'input': self.encodeText(text, learn),
        #     'output': self.intentMap.get(intent),
        # }
    
    def encodeCorpus(self, corpus):
        result = {'train': [], 'validation': []}
        for item in corpus:
            if 'utterances' in item and item['utterances']:
                for utterance in item['utterances']:
                    result['train'].append(self.encode(utterance, item['intent'], True))
        for item in corpus:
            if 'tests' in item and item['tests']:
                for test in item['tests']:
                    result['validation'].append(self.encode(test, item['intent']))
        return result
