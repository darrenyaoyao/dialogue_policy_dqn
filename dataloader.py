'''
Load all the data and split them into train and val.
Each data is look like this (all in tuple):
(st, at, st1, rt, terminal)
st: (goal, song, singer, album)
at: (act, song, singer, album)
rt: int
terminal: int (1 for true, 0 for false)

Usage:
    1. first initialize the class ,specify the data set filepath
       loader = Dataloader(filepath)
    2. Split the data to training set and validation set, specify the proportion of validate to whole set
       loader.split(0.2)
    3. get training batch
       loader.get_train_batch(64)
       return list of (st, at, st1, rt, terminal)

'''
import numpy as np
import pickle, copy
import random, math


class Dataloader():
    def __init__(self, filename='training_data_encode.npy'):
        self.data = np.load(filename)
        self.mapping = pickle.load(open('mapping.p', 'rb'))
        self.goal_size = len(self.mapping['goal'])
        self.action_size = len(self.mapping['action'])
        self.song_size = len(self.mapping['song'])
        self.singer_size = len(self.mapping['singer'])
        self.album_size = len(self.mapping['album'])
        self.dataset = []
        self.val_set = []
        self.train_set = []
        self.makeSet()
    #def chunk(self,l,n):

    def get_train_batch(self, batch_size):
        batch = []
        for i in range(batch_size):
            batch.append(random.choice(self.train_set))
        return batch

    def get_val(self):
        return self.val_set

    def split(self, ratio):
        random.shuffle(self.dataset)
        #print("math.floor = ", math.floor(ratio*len(self.dataset)))
        self.val_set = self.dataset[:int(math.floor(ratio*len(self.dataset)))]
        self.train_set = self.dataset[int(math.floor(ratio*len(self.dataset))):]

    def getState(self, bot_state):
        state = list(bot_state)
        del state[-1]
        del state[-1]
        return tuple(state)

    def getAction(self, bot_action):
        action = list(bot_action)
        del action[-1]
        del action[-1]
        return tuple(action)

    def makeSet(self):
        for conversations in self.data:
            reward = copy.copy(conversations[-1])
            del conversations[-1]
            for index, turn in enumerate(conversations):
                state = self.getState(turn['Bot_state'])
                action = self.getAction(turn['Bot_action'])
                if index == len(conversations)-1:
                    #print(reward)
                    re = reward
                    next_state = (0, 0, 0, 0)
                    terminal = 1
                else:
                    re = 0
                    next_state = self.getState(conversations[index+1]['Bot_state'])
                    terminal = 0

                self.dataset.append((state, action, next_state, re, terminal))


if __name__ == "__main__":
    loader = Dataloader()
    #loader.makeSet()
    print(loader.dataset[:10])
    print("len = ", len(loader.dataset))
    loader.Split(0.1)
    print("get the bactch size = ",len(loader.get_train_batch(64)))
    def check(data):
        for d in data:
            if type(d[0]) != tuple or type(d[1]) != tuple:
                print("error!")


