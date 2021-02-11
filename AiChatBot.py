"# AiChatBot"

from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy as np
import random
import json

import PreprocessingUnit as pu
import TrainigUnit as tu


def loadNetwork():
    with open(tu.trainingIntentsPath) as file:
        data = json.load(file)
        words, labels, training, output = tu.loadModels()
        model = tu.defineModel(training, output)
        model.load(tu.modelPath)
        return model, data, words, labels, training, output




def chat():
    print("Start talking:")
    while True:
        inp = input("You : ")
        if inp.lower() == ":q":
            break
        result = model.predict([pu.encriptInput(inp, words)])[0]
        result_index = np.argmax(result)
        tag = labels[result_index]
        # print(result[result_index] )
        if result[result_index] > 0.8:

            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg["responses"]

            print(random.choice(responses))
        else:
            print("Sorry , i was not prepared for that , please ask something else")


if __name__ == '__main__':
    model, data, words, labels, training, output = loadNetwork()
    chat()
    # print("model activated");