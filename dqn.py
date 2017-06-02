import tensorflow as tf


class DQN(object):
    GAMMA = 0.9

    def __init__(self, goal_size=10, act_size=10, song_size=10,
                 singer_size=10, album_size=10):
        self.state_goal = tf.placeholder(tf.int32,
                                         [None], name="state_goal")
        self.state_song = tf.placeholder(tf.int32,
                                         [None], name="state_song")
        self.state_singer = tf.placeholder(tf.int32,
                                           [None], name="state_singer")
        self.state_album = tf.placeholder(tf.int32,
                                          [None], name="state_album")
        self.action_act = tf.placeholder(tf.int32,
                                         [None], name="action_act")
        self.action_song = tf.placeholder(tf.int32,
                                          [None], name="action_song")
        self.action_singer = tf.placeholder(tf.int32,
                                            [None], name="action_singer")
        self.action_album = tf.placeholder(tf.int32,
                                           [None], name="action_album")
        self.target_q = tf.placeholder(tf.float32,
                                       [None], name="target_q")
        self.reward = tf.placeholder(tf.float32,
                                     [None], name="reward")
        self.terminal = tf.placeholder(tf.float32,
                                       [None], name="terminal")

        with tf.name_scope("embedding"):
            self.goal_w = tf.Variable(tf.random_uniform([goal_size, 2],
                                                        name="goal_w"))
            self.act_w = tf.Variable(tf.random_uniform([act_size, 4],
                                                       name="act_w"))
            self.song_w = tf.Variable(tf.random_uniform([song_size, 10],
                                                        name="song_w"))
            self.singer_w = tf.Variable(tf.random_uniform([singer_size, 10],
                                                          name="singer_w"))
            self.album_w = tf.Variable(tf.random_uniform([album_size, 10],
                                                         name="album_w"))
            self.st_goal = tf.nn.embedding_lookup(self.goal_w, self.state_goal)
            self.st_song = tf.nn.embedding_lookup(self.song_w, self.state_song)
            self.st_singer = tf.nn.embedding_lookup(self.singer_w,
                                                    self.state_singer)
            self.st_album = tf.nn.embedding_lookup(self.album_w,
                                                   self.state_album)
            self.at_act = tf.nn.embedding_lookup(self.act_w, self.action_act)
            self.at_song = tf.nn.embedding_lookup(self.song_w,
                                                  self.action_song)
            self.at_singer = tf.nn.embedding_lookup(self.singer_w,
                                                    self.action_singer)
            self.at_album = tf.nn.embedding_lookup(self.album_w,
                                                   self.action_album)
            self.st = tf.concat([self.st_goal, self.st_song,
                                 self.st_singer, self.st_album], 1)  # 32 dim
            self.at = tf.concat([self.at_act, self.at_song,
                                 self.at_singer, self.at_album], 1)  # 34 dim

        with tf.name_scope("MLP"):
            w0_st = tf.Variable(
                tf.truncated_normal([32, 64], stddev=0.1), name="w0_st")
            b0_st = tf.Variable(tf.constant(0.1, shape=[64]), name="b0_st")
            h0_st = tf.nn.relu(tf.nn.xw_plus_b(self.st, w0_st, b0_st))
            w0_at = tf.Variable(
                tf.truncated_normal([34, 64], stddev=0.1), name="w0_at")
            b0_at = tf.Variable(tf.constant(0.1, shape=[64]), name="b0_at")
            h0_at = tf.nn.relu(tf.nn.xw_plus_b(self.at, w0_at, b0_at))
            h0 = tf.concat([h0_st, h0_at], 1)
            w1 = tf.Variable(
                tf.truncated_normal([128, 1], stddev=0.1), name="w1")
            b1 = tf.Variable(tf.constant(0.1, shape=[1]), name="b1")
            h1 = tf.nn.relu(tf.nn.xw_plus_b(h0, w1, b1))
            self.q = tf.reshape(h1, [-1], name="q")

        with tf.name_scope("target"):
            self.target = self.reward + \
                self.GAMMA * tf.multiply(self.terminal, self.target_q)

        with tf.name_scope("loss"):
            self.loss = tf.reduce_sum(tf.pow(self.target-self.q, 2))
