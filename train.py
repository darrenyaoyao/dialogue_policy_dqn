import tensorflow as tf
import numpy as np
import random
from copy import deepcopy
import datetime
from dqn import DQN

memory = []

with tf.Graph().as_default():
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True
    ))
    with sess.as_default():
        dqn = DQN()

        # Define Training Procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(dqn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars,
                                             global_step=global_step)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def max_q(at_batch, st1_batch, action_size):
            tmp = []
            all_action = []
            all_st1 = []
            for a in at_batch:
                for i in range(8):
                    a0 = deepcopy(a)
                    idx_1 = i % 2
                    idx_2 = (i / 2) % 2
                    idx_3 = (i / 4) % 2
                    if idx_1 == 0:
                        a0[1] = 0
                    if idx_2 == 0:
                        a0[2] = 0
                    if idx_3 == 0:
                        a0[3] = 0
                    tmp.append(a0)
            for a in tmp:
                for i in range(action_size):
                    a0 = deepcopy(a)
                    a[0] = i
                    all_action.append(a)
            for s in st1_batch:
                for i in range(8*action_size):
                    all_st1.append(s)
            st_goal_batch = [s[0] for s in all_st1]
            st_song_batch = [s[1] for s in all_st1]
            st_singer_batch = [s[2] for s in all_st1]
            st_album_batch = [s[3] for s in all_st1]
            at_act_batch = [a[0] for a in all_action]
            at_song_batch = [a[1] for a in all_action]
            at_singer_batch = [a[2] for a in all_action]
            at_album_batch = [a[3] for a in all_action]
            feed_dict = {
                dqn.state_goal: st_goal_batch,
                dqn.state_song: st_song_batch,
                dqn.state_singer: st_singer_batch,
                dqn.state_album: st_album_batch,
                dqn.action_act: at_act_batch,
                dqn.action_song: at_song_batch,
                dqn.action_singer: at_singer_batch,
                dqn.action_album: at_album_batch
            }
            q = sess.run(
                [dqn.q],
                feed_dict
            )
            max_q = []
            for i in range(0, len(q), 8*action_size):
                q_max = float('-inf')
                for j in range(8*action_size):
                    if q[i+j] > q_max:
                        q_max = q[i+j]
                max_q.append(q_max)
            return max_q

        def train_step(st_batch, at_batch, target_q_batch,
                       rt_batch, terminal_batch):
            st_goal_batch = [s[0] for s in st_batch]
            st_song_batch = [s[1] for s in st_batch]
            st_singer_batch = [s[2] for s in st_batch]
            st_album_batch = [s[3] for s in st_batch]
            at_act_batch = [a[0] for a in at_batch]
            at_song_batch = [a[1] for a in at_batch]
            at_singer_batch = [a[2] for a in at_batch]
            at_album_batch = [a[3] for a in at_batch]
            feed_dict = {
                dqn.state_goal: st_goal_batch,
                dqn.state_song: st_song_batch,
                dqn.state_singer: st_singer_batch,
                dqn.state_album: st_album_batch,
                dqn.action_act: at_act_batch,
                dqn.action_song: at_song_batch,
                dqn.action_singer: at_singer_batch,
                dqn.action_album: at_album_batch,
                dqn.target_q: target_q_batch,
                dqn.reward: rt_batch,
                dqn.terminal: terminal_batch
            }
            _, step, loss = sess.run(
                [train_op, global_step, dqn.loss],
                feed_dict)

        def dev_step(st_batch, at_batch, target_q_batch,
                     rt_batch, terminal_batch):
            st_goal_batch = [s[0] for s in st_batch]
            st_song_batch = [s[1] for s in st_batch]
            st_singer_batch = [s[2] for s in st_batch]
            st_album_batch = [s[3] for s in st_batch]
            at_act_batch = [a[0] for a in at_batch]
            at_song_batch = [a[1] for a in at_batch]
            at_singer_batch = [a[2] for a in at_batch]
            at_album_batch = [a[3] for a in at_batch]
            feed_dict = {
                dqn.state_goal: st_goal_batch,
                dqn.state_song: st_song_batch,
                dqn.state_singer: st_singer_batch,
                dqn.state_album: st_album_batch,
                dqn.action_act: at_act_batch,
                dqn.action_song: at_song_batch,
                dqn.action_singer: at_singer_batch,
                dqn.action_album: at_album_batch,
                dqn.target_q: target_q_batch,
                dqn.reward: rt_batch,
                dqn.terminal: terminal_batch
            }
            step, loss = sess.run(
                [global_step, dqn.loss],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}".format(time_str, step, loss))

        while True:
            st_batch, at_batch, st1_batch, rt_batch, terminal_batch \
                 = dataloader.get_train_batch(64)
            target_q_batch = max_q(at_batch, st1_batch)
            train_step(st_batch, at_batch, target_q_batch,
                       rt_batch, terminal_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                dev_target_q_batch = max_q(dev_at_batch, dev_st1_batch)
                dev_step(dev_st_batch, dev_at_batch, dev_target_q_batch,
                         dev_rt_batch, dev_terminal_batch)

