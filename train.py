import tensorflow as tf
import numpy as np
import random
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

        def max_q(at_batch, st1_batch):
            

        def train_step(st_batch, at_batch, target_q_batch,
                       rt_batch, terminal_batch):
            st_goal_batch, st_song_batch, st_singer_batch, st_album_batch \
                = st_batch
            at_act_batch, at_song_batch, at_singer_batch, at_album_batch \
                = at_batch
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
            st_goal_batch, st_song_batch, st_singer_batch, st_album_batch \
                = st_batch
            at_act_batch, at_song_batch, at_singer_batch, at_album_batch \
                = at_batch
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
                 = random.sample(memory, batch_size)
            target_q_batch = max_q(at_batch, st1_batch)

