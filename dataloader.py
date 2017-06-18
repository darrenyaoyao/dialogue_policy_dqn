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
    def __init__(self, filename='data.npy'):
        self.data = np.load(filename)
        self.training_data = []
        self.all_states = np.load('all_states.npy')
        self.all_actions = np.load('all_actions.npy')
        self.all_intents = np.load('all_intents.npy')
        self.goal_size = len(self.all_states)
        self.action_size = len(self.all_actions)
        self.intent_size = len(self.all_intents)
        self.states_dict = {}
        self.actions_dict = {}
        self.intents_dict = {}
        for i in range(len(self.all_states)):
            self.states_dict[self.all_states[i]] = i
        for i in range(len(self.all_actions)):
            self.actions_dict[self.all_actions[i]] = i
        for i in range(len(self.all_intents)):
            self.intents_dict[self.all_intents[i]] = i
        self.inv_states_dict = {v: k for k, v in self.states_dict.iteritems()}
        self.inv_actions_dict = {v: k for k, v in self.actions_dict.iteritems()}
        self.inv_intents_dict = {v: k for k, v in self.intents_dict.iteritems()}
        for d in self.data:
            training_d = {}
            training_d['rt'] = d['rt']
            training_d['terminal'] = d['terminal']
            training_d['at'] = self.actions_dict[d['at']]
            # state
            st = []
            st.append(self.states_dict[d['st'][0]])
            if d['st'][1][0]:
                st.append([1])
            else:
                st.append([0])
            if d['st'][2][0]:
                st.append([1])
            else:
                st.append([0])
            if d['st'][3][0]:
                st.append([1])
            else:
                st.append([0])
            training_d['st'] = st
            # state1
            st_1 = []
            st_1.append(self.states_dict[d['st_1'][0]])
            if d['st_1'][1][0][0]:
                st_1.append([1])
            else:
                st_1.append([0])
            if d['st_1'][2][0][0]:
                st_1.append([1])
            else:
                st_1.append([0])
            if d['st_1'][3][0][0]:
                st_1.append([1])
            else:
                st_1.append([0])
            training_d['st_1'] = st_1
            # history
            histories = []
            for h in d['history']:
                history = []
                history.append(self.intents_dict[h[0]])
                if h[1]:
                    history.append([1])
                else:
                    history.append([0])
                if h[2]:
                    history.append([1])
                else:
                    history.append([0])
                if h[3]:
                    history.append([1])
                else:
                    history.append([0])
                histories.append(history)
            if len(histories) >= 4:
                training_d['history'] = histories[-4:]
            else:
                n = 4 - len(histories)
                training_d['history'] = histories
                for i in range(n):
                    training_d['history'].append([self.intents_dict[''], [0], [0], [0]])
            # history_1
            histories = []
            for h in d['history_1']:
                history = []
                history.append(self.intents_dict[h[0]])
                if h[1]:
                    history.append([1])
                else:
                    history.append([0])
                if h[2]:
                    history.append([1])
                else:
                    history.append([0])
                if h[3]:
                    history.append([1])
                else:
                    history.append([0])
                histories.append(history)
            if len(histories) >= 4:
                training_d['history_1'] = histories[-4:]
            else:
                n = 4 - len(histories)
                training_d['history_1'] = histories
                for i in range(n):
                    training_d['history_1'].append([self.intents_dict[''], [0], [0], [0]])
            self.training_data.append(training_d)
        self.val_data = self.training_data[:100]
        self.training_data = self.training_data[100:]

    def get_train_batch(self, batch_size):
        batch = []
        for i in range(batch_size):
            batch.append(random.choice(self.training_data))
        return batch


if __name__ == "__main__":
    dataloader = Dataloader()
