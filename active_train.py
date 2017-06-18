import tensorflow as tf
from copy import deepcopy
from dataloader import Dataloader
from Data.DST import DialogStateTracker
from Data.eval_multi_task_rnn import Eval
from Data.UserSimulator import UserSimulator
import datetime
from dqn import DQN


tf.app.flags.DEFINE_integer("evaluate_every", 100,
                            "Number of step for model evaluation.")
tf.app.flags.DEFINE_integer("batch_size", 64, "batch size for training")

FLAGS = tf.app.flags.FLAGS


def train():
    dataloader = Dataloader()
    LU = Eval()
    DST = DialogStateTracker()
    USERSIM = UserSimulator()
    val_data = dataloader.val_data
    with tf.Graph().as_default():
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True
        ))
        with sess.as_default():
            dqn = DQN(goal_size=dataloader.goal_size,
                      act_size=dataloader.action_size,
                      intent_size=dataloader.intent_size)

            # Define Training Procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(dqn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars,
                                                 global_step=global_step)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # predict the max Q of st1
            def max_q(batch):
                action_size = dataloader.action_size
                st1_goal_batch = [d['st_1'][0] for d in batch for i in range(action_size)]
                st1_song_batch = [d['st_1'][1] for d in batch for i in range(action_size)]
                st1_singer_batch = [d['st_1'][2] for d in batch for i in range(action_size)]
                st1_album_batch = [d['st_1'][3] for d in batch for i in range(action_size)]
                history0_intent = [d['history_1'][0][0] for d in batch for i in range(action_size)]
                history0_song = [d['history_1'][0][1] for d in batch for i in range(action_size)]
                history0_singer = [d['history_1'][0][2] for d in batch for i in range(action_size)]
                history0_album = [d['history_1'][0][3] for d in batch for i in range(action_size)]
                history1_intent = [d['history_1'][1][0] for d in batch for i in range(action_size)]
                history1_song = [d['history_1'][1][1] for d in batch for i in range(action_size)]
                history1_singer = [d['history_1'][1][2] for d in batch for i in range(action_size)]
                history1_album = [d['history_1'][1][3] for d in batch for i in range(action_size)]
                history2_intent = [d['history_1'][2][0] for d in batch for i in range(action_size)]
                history2_song = [d['history_1'][2][1] for d in batch for i in range(action_size)]
                history2_singer = [d['history_1'][2][2] for d in batch for i in range(action_size)]
                history2_album = [d['history_1'][2][3] for d in batch for i in range(action_size)]
                history3_intent = [d['history_1'][3][0] for d in batch for i in range(action_size)]
                history3_song = [d['history_1'][3][1] for d in batch for i in range(action_size)]
                history3_singer = [d['history_1'][3][2] for d in batch for i in range(action_size)]
                history3_album = [d['history_1'][3][3] for d in batch for i in range(action_size)]
                at_batch = [j for i in range(len(batch)) for j in range(action_size)]
                feed_dict = {
                    dqn.state_goal: st1_goal_batch,
                    dqn.state_song: st1_song_batch,
                    dqn.state_singer: st1_singer_batch,
                    dqn.state_album: st1_album_batch,
                    dqn.action: at_batch,
                    dqn.history_intent[0]: history0_intent,
                    dqn.history_intent[1]: history1_intent,
                    dqn.history_intent[2]: history2_intent,
                    dqn.history_intent[3]: history3_intent,
                    dqn.history_song[0]: history0_song,
                    dqn.history_song[1]: history1_song,
                    dqn.history_song[2]: history2_song,
                    dqn.history_song[3]: history3_song,
                    dqn.history_singer[0]: history0_singer,
                    dqn.history_singer[1]: history1_singer,
                    dqn.history_singer[2]: history2_singer,
                    dqn.history_singer[3]: history3_singer,
                    dqn.history_album[0]: history0_album,
                    dqn.history_album[1]: history1_album,
                    dqn.history_album[2]: history2_album,
                    dqn.history_album[3]: history3_album,
                }
                q = sess.run(
                    dqn.q,
                    feed_dict
                )
                max_q = []
                q_actions = []

                def valid(i, j):
                    num = i + j
                    if j in [0, 5, 6, 8, 11]:
                        return True
                    elif j in [1, 2, 3, 9, 10] and st1_singer_batch[num] == 0:
                        return False
                    elif j in [3, 4, 7, 10, 13] and st1_song_batch[num] == 0:
                        return False
                    elif j in [9, 12] and st1_album_batch[num] == 0:
                        return False
                    else:
                        return True

                for i in range(0, len(q), action_size):
                    q_max = float('-inf')
                    q_action = 0
                    for j in range(action_size):
                        if q[i+j] > q_max and valid(i, j):
                            q_max = q[i+j]
                            q_action = j
                    max_q.append(q_max)
                    q_actions.append(q_action)
                return max_q, q_actions

            def train_step(batch, target_q):
                feed_dict = batch_form(batch, target_q)
                _, step, loss, q = sess.run(
                    [train_op, global_step, dqn.loss, dqn.q],
                    feed_dict)
                '''
                print("Target")
                print(target_q)
                print("Q")
                print(q)
                '''

            def dev_step(batch, target_q):
                feed_dict = batch_form(batch, target_q)
                step, loss = sess.run(
                    [global_step, dqn.loss],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}".format(time_str, step, loss))

            def batch_form(batch, target_q):
                st_goal_batch = [d['st'][0] for d in batch]
                st_song_batch = [d['st'][1] for d in batch]
                st_singer_batch = [d['st'][2] for d in batch]
                st_album_batch = [d['st'][3] for d in batch]
                at_batch = [d['at'] for d in batch]
                rt_batch = [d['rt'] for d in batch]
                terminal_batch = [d['terminal'] for d in batch]
                history0_intent = [d['history'][0][0] for d in batch]
                history0_song = [d['history'][0][1] for d in batch]
                history0_singer = [d['history'][0][2] for d in batch]
                history0_album = [d['history'][0][3] for d in batch]
                history1_intent = [d['history'][1][0] for d in batch]
                history1_song = [d['history'][1][1] for d in batch]
                history1_singer = [d['history'][1][2] for d in batch]
                history1_album = [d['history'][1][3] for d in batch]
                history2_intent = [d['history'][2][0] for d in batch]
                history2_song = [d['history'][2][1] for d in batch]
                history2_singer = [d['history'][2][2] for d in batch]
                history2_album = [d['history'][2][3] for d in batch]
                history3_intent = [d['history'][3][0] for d in batch]
                history3_song = [d['history'][3][1] for d in batch]
                history3_singer = [d['history'][3][2] for d in batch]
                history3_album = [d['history'][3][3] for d in batch]
                return {
                    dqn.state_goal: st_goal_batch,
                    dqn.state_song: st_song_batch,
                    dqn.state_singer: st_singer_batch,
                    dqn.state_album: st_album_batch,
                    dqn.action: at_batch,
                    dqn.reward: rt_batch,
                    dqn.terminal: terminal_batch,
                    dqn.history_intent[0]: history0_intent,
                    dqn.history_intent[1]: history1_intent,
                    dqn.history_intent[2]: history2_intent,
                    dqn.history_intent[3]: history3_intent,
                    dqn.history_song[0]: history0_song,
                    dqn.history_song[1]: history1_song,
                    dqn.history_song[2]: history2_song,
                    dqn.history_song[3]: history3_song,
                    dqn.history_singer[0]: history0_singer,
                    dqn.history_singer[1]: history1_singer,
                    dqn.history_singer[2]: history2_singer,
                    dqn.history_singer[3]: history3_singer,
                    dqn.history_album[0]: history0_album,
                    dqn.history_album[1]: history1_album,
                    dqn.history_album[2]: history2_album,
                    dqn.history_album[3]: history3_album,
                    dqn.target_q: target_q
                }

            def msg_action(action, song, singer, album):
                if action == 0:
                    return '嗨我是音樂小儂，有什麼能為您效勞嗎？'
                if action == 1:
                    return '請問演唱者是{}'+singer+'{}嗎？'
                if action == 2:
                    return '您要找{}'+singer+'{}的什麼歌的歌詞？'
                if action == 3:
                    return '以下是{}'+singer+'{}唱的{}'+song+'{}'+'的歌詞，如果您滿意這次的服務，對我說謝謝我會很開心。'
                if action == 4:
                    return '請問您要誰唱的{}'+song+'{}呢？'
                if action == 5:
                    return '不好意思可以再說一次嗎？'
                if action == 6:
                    return '不好意思，我不太了解您的意思，可以再說一遍嗎？'
                if action == 7:
                    return '您是要聽{}'+song+'{}嗎？'
                if action == 8:
                    return '請問您要聽什麼歌呢？'
                if action == 9:
                    return '您詢問的專輯{}'+album+'{}的演唱者爲{}'+singer+'{}'
                if action == 10:
                    return '為您播放{}'+singer+'{}的{}'+song+'{}，如果您滿意這次的服務，對我說謝謝我會很開心。'
                if action == 11:
                    return '您有沒有特定的喜好呢？'
                if action == 12:
                    return '您是要找唱專輯{}'+album+'{}的歌手嗎？'
                if action == 13:
                    return '很抱歉，您詢問的歌曲{}'+song+'{}的演唱者在我們系統沒有辦法被找到'


            def conversation(Iter):
                MAX_TURN = 10
                restart_intent = ['thankyou', 'bye', 'restart']
                bye = ['謝謝', '謝謝您', '再見', '再會', '掰掰', 'bye', '88']
                iteration = 0
                total_reward = 0
                succes_rate = 0
                while(iteration < Iter):
                    print(iteration)
                    Action = "welcomemsg"
                    Slotsong = ''
                    Slotsinger = ''
                    Slotalbum = ''
                    Goal, Song, Singer, Album = '', '', '', ''
                    counter = 0
                    user_response = ''
                    Bot_msg = '嗨我是音樂小儂，有什麼能為您效勞嗎？'
                    history = []
                    while (counter < MAX_TURN and user_response not in bye):
                        user_response = USERSIM.feed(Bot_msg)
                        LU_output = LU.feed_sentence(user_response)
                        DST_feedin = str(LU_output[0]+","+LU_output[1].encode('utf8')+","+LU_output[2].encode('utf8')+","+LU_output[3].encode('utf8')+","+LU_output[4].encode('utf8')+","+LU_output[5].encode('utf8'))
                        if (DST_feedin.split(','))[0] in restart_intent:
                            DST.erase_slot(True,True,True,True,True,True,True)
                            DST_feedin = ",,,,,"
                        LU_container = tuple(DST_feedin.split(','))
                        history0 = history[:]
                        history.append(LU_container)
                        goal0 = Goal
                        song0 = Song
                        singer0 = Singer
                        album0 = Album
                        action0 = Action
                        Goal, Song, Singer, Album, _, _ = \
                            DST.feed(DST_feedin, Action, Slotsong,
                                     Slotsinger, Slotalbum, "", "")
                        feed_data = dataloader.generate_data(history, [Goal, Song, Singer, Album])
                        _, Action = max_q([feed_data])
                        Bot_msg = msg_action(Action, Song, Singer, Album)
                        counter += 1
                        if counter < MAX_TURN and user_response not in bye:
                            dataloader.save_data(history0, [goal0, song0, singer0, album0],
                                                 history, [Goal, Song, Singer, Album], action0, 0, 0)
                        else:
                            total_reward += USERSIM.get_reward()
                            if USERSIM.get_reward() > 0:
                                succes_rate += 1
                            dataloader.save_data(history0, [goal0, song0, singer0, album0],
                                                 history, [Goal, Song, Singer, Album], action0, USERSIM.get_reward(), 1)
                return succes_rate/Iter, total_reward/Iter

            output_file = open("result", 'w')
            current_step = 0
            while True:
                success, reward = conversation(100)
                print(str(current_step)+" "+str(success)+" "+str(reward)+"\n")
                output_file.write(str(current_step)+" "+str(success)+" "+str(reward)+"\n")
                for i in range(1000):
                    train_batch = dataloader.get_train_batch(FLAGS.batch_size)
                    train_target_q_batch, q_actions_batch = max_q(train_batch)
                    train_step(train_batch, train_target_q_batch)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % FLAGS.evaluate_every == 0:
                        dev_target_q_batch, dev_q_actions_batch = max_q(val_data)
                        #print(val_data[:10])
                        #print(dev_q_actions_batch[:10])
                        dev_step(val_data, dev_target_q_batch)


def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()
