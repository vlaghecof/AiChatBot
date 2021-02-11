from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import json
import tflearn
from tensorflow.python.framework import ops
import pickle
import PreprocessingUnit  as pu

modelPath=r"Model/model.tflearn"
dataPath="Data/data.pickler"
trainingIntentsPath="Intents/intents.json"


def loadModels():
    with open(dataPath, "rb") as f:
        words, labels, training, output = pickle.load(f)
    return words, labels, training, output


def storeModels(words, labels, training, output):
    with open(dataPath, "wb") as f:
        pickle.dump((words, labels, training, output), f)


def defineModel(training, output):
    ops.reset_default_graph()

    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)
    return model


def generateModel(intentsFile):
    with open(intentsFile) as file:
        data = json.load(file)

        words, labels, docs_x, docs_y = pu.defineDataSet(data)
        words = [stemmer.stem(w.lower()) for w in words if w not in ["?", ".", "!"]]
        words = sorted(list(set(words)))
        labels = sorted(labels)

        training, output = pu.defineTrainingData(docs_x, docs_y, labels, words)
        storeModels(words, labels, training, output)
        model = defineModel(training, output)
        model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
        model.save(modelPath)
        return model, words, labels



if __name__ == '__main__':
    generateModel(trainingIntentsPath)