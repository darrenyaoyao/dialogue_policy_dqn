import tensorflow as tf


class DQN(object):
    GAMMA = 0.9

    def __init__(self, goal_size=10, act_size=10, intent_size=10):
        self.state_goal = tf.placeholder(tf.int32,
                                         [None], name="state_goal")
        self.state_song = tf.placeholder(tf.float32,
                                         [None, 1], name="state_song")
        self.state_singer = tf.placeholder(tf.float32,
                                           [None, 1], name="state_singer")
        self.state_album = tf.placeholder(tf.float32,
                                           [None, 1], name="state_singer")
        self.action = tf.placeholder(tf.int32,
                                     [None], name="action")
        self.history_intent = []
        self.history_song = []
        self.history_singer = []
        self.history_album = []
        for i in range(4):
            self.history_intent.append(
                    tf.placeholder(tf.int32,
                                   [None],
                                   name="history{}_intent".format(str(i))))
            self.history_song.append(
                    tf.placeholder(tf.float32,
                                   [None, 1],
                                   name="history{}_song".format(str(i))))
            self.history_singer.append(
                    tf.placeholder(tf.float32,
                                   [None, 1],
                                   name="history{}_singer".format(str(i))))
            self.history_album.append(
                    tf.placeholder(tf.float32,
                                   [None, 1],
                                   name="history{}_album".format(str(i))))
        self.target_q = tf.placeholder(tf.float32,
                                       [None], name="target_q")
        self.reward = tf.placeholder(tf.float32,
                                     [None], name="reward")
        self.terminal = tf.placeholder(tf.float32,
                                       [None], name="terminal")

        # one hot embedding for each slot and goal, action
        with tf.name_scope("embedding"):
            self.goal_w = tf.Variable(tf.random_uniform([goal_size, 2],
                                                        name="goal_w"))
            self.intent_w = tf.Variable(tf.random_uniform([intent_size, 2],
                                                          name="intent_w"))
            self.act_w = tf.Variable(tf.random_uniform([act_size, 2],
                                                       name="act_w"))
            self.st_goal = tf.nn.embedding_lookup(self.goal_w, self.state_goal)
            self.at_act = tf.nn.embedding_lookup(self.act_w, self.action)
            self.hs_intent = []
            for i in range(4):
                self.hs_intent.append(
                    tf.nn.embedding_lookup(self.intent_w, self.history_intent[i])
                )
            state = [self.st_goal, self.state_song, self.state_singer, self.state_album]
            for i in range(4):
                state.append(self.hs_intent[i])
                state.append(self.history_song[i])
                state.append(self.history_singer[i])
                state.append(self.history_album[i])
            self.st = tf.concat(state, 1) # dim 25
            self.at = self.at_act # dim 2

        #  model architecture
        with tf.name_scope("MLP"):
            w0_st = tf.Variable(
                tf.truncated_normal([25, 50], stddev=0.1), name="w0_st")
            b0_st = tf.Variable(tf.constant(0.1, shape=[50]), name="b0_st")
            h0_st = tf.nn.relu(tf.nn.xw_plus_b(self.st, w0_st, b0_st))
            w0_at = tf.Variable(
                tf.truncated_normal([2, 4], stddev=0.1), name="w0_at")
            b0_at = tf.Variable(tf.constant(0.1, shape=[4]), name="b0_at")
            h0_at = tf.nn.tanh(tf.nn.xw_plus_b(self.at, w0_at, b0_at))
            h0 = tf.concat([h0_st, h0_at], 1)
            w1 = tf.Variable(
                tf.truncated_normal([54, 1], stddev=0.1), name="w1")
            b1 = tf.Variable(tf.constant(0.1, shape=[1]), name="b1")
            h1 = tf.nn.tanh(tf.nn.xw_plus_b(h0, w1, b1))
            self.q = tf.reshape(h1, [-1], name="q")*40

        # target reward
        with tf.name_scope("target"):
            self.target = self.reward + \
                self.GAMMA * tf.multiply(self.terminal, self.target_q)

        # loss function
        with tf.name_scope("loss"):
            self.loss = tf.reduce_sum(tf.pow(self.target-self.q, 2))
