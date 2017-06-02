import tensorflow as tf
from copy import deepcopy
from dataloader import Dataloader
import datetime
from dqn import DQN


tf.app.flags.DEFINE_integer("evaluate_every", 100,
                            "Number of step for model evaluation.")
tf.app.flags.DEFINE_integer("batch_size", 64, "batch size for training")

FLAGS = tf.app.flags.FLAGS


def train():
    dataloader = Dataloader()
    dataloader.split(0.1)
    val_data = dataloader.get_val()
    dev_st_batch = [d[0] for d in val_data]
    dev_at_batch = [d[1] for d in val_data]
    dev_st1_batch = [d[2] for d in val_data]
    dev_rt_batch = [d[3] for d in val_data]
    dev_terminal_batch = [d[4] for d in val_data]
    with tf.Graph().as_default():
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True
        ))
        with sess.as_default():
            dqn = DQN(goal_size=dataloader.goal_size,
                      act_size=dataloader.action_size,
                      song_size=dataloader.song_size,
                      singer_size=dataloader.singer_size,
                      album_size=dataloader.album_size)

            # Define Training Procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(dqn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars,
                                                 global_step=global_step)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # predict the max Q of st1
            def max_q(st_batch, st1_batch, action_size=9):
                tmp = []
                all_action = []
                all_st1 = []
                # generate all possible action
                for s in st_batch:
                    for i in range(8):
                        a0 = list(deepcopy(s))
                        idx_1 = i % 2
                        idx_2 = (i / 2) % 2
                        idx_3 = (i / 4) % 2
                        if idx_1 == 0:
                            a0[1] = 0
                        if idx_2 == 0:
                            a0[2] = 0
                        if idx_3 == 0:
                            a0[3] = 0
                        tmp.append(tuple(a0))
                for a in tmp:
                    for i in range(action_size):
                        a0 = list(deepcopy(a))
                        a0[0] = i
                        all_action.append(tuple(a0))
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
                    dqn.q,
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
                print(rt_batch)
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
                train_batch = dataloader.get_train_batch(FLAGS.batch_size)
                st_batch = [d[0] for d in train_batch]
                at_batch = [d[1] for d in train_batch]
                st1_batch = [d[2] for d in train_batch]
                rt_batch = [d[3] for d in train_batch]
                terminal_batch = [d[4] for d in train_batch]
                target_q_batch = max_q(st_batch, st1_batch)
                train_step(st_batch, at_batch, target_q_batch,
                           rt_batch, terminal_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    dev_target_q_batch = max_q(dev_at_batch, dev_st1_batch)
                    dev_step(dev_st_batch, dev_at_batch, dev_target_q_batch,
                             dev_rt_batch, dev_terminal_batch)


def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()
