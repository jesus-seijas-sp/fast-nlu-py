import json
from src import Neural

with open('benchmark/corpus-massive-en.json') as f:
    corpus = json.load(f)

net = Neural({ "log": True })
net.train(corpus["data"])
