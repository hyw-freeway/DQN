import argparse
import os
import shutil
from random import random, randint, sample

import numpy as np
import paddle
import paddle.nn as nn
from visualdl import LogWriter
# import paddle.fluid as fluid
from src.deep_q_network import DeepQNetwork
from src.tetris import Tetris
from collections import deque
# 一、导入分布式专用 Fleet API
from paddle.distributed import fleet
# 构建分布式数据加载器所需 API
from paddle.io import DataLoader, DistributedBatchSampler
# 设置 GPU 环境
paddle.set_device('gpu:0')

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=2000)
    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--replay_memory_size", type=int, default=30000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="vdllog")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


def train(opt):
    

    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)

    # 二、初始化 Fleet 环境
    fleet.init(is_collective=True)
    
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    model = DeepQNetwork()
    # 三、构建分布式训练使用的网络模型
    model = fleet.distributed_model(model)
    lr = opt.lr
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=lr)
    # 四、构建分布式训练使用的优化器
    opt = fleet.distributed_optimizer(opt)
    criterion = nn.MSELoss()

    state = env.reset()

    replay_memory = deque(maxlen=opt.replay_memory_size)
    epoch = 0
    with LogWriter(logdir=opt.log_path) as writer:
        while epoch < opt.num_epochs:
            next_steps = env.get_next_states()
            # Exploration or exploitation
            epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                    opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
            u = random()
            random_action = u <= epsilon
            next_actions, next_states = zip(*next_steps.items())
            next_states = paddle.stack(next_states)


            if random_action:
                index = randint(0, len(next_steps) - 1)
            else:
                model.eval()
                with paddle.no_grad():
                    predictions = model(next_states)[:, 0]
                model.train()
                index = paddle.argmax(predictions).item()

            next_state = next_states[index, :]
            action = next_actions[index]

            reward, done = env.step(action, render=True)

            replay_memory.append([state, reward, next_state, done])
            if done:
                final_score = env.score
                final_tetrominoes = env.tetrominoes
                final_cleared_lines = env.cleared_lines
                state = env.reset()

            else:
                state = next_state
                continue
            if len(replay_memory) < opt.replay_memory_size / 10:
                print(len(replay_memory)/(opt.replay_memory_size / 10))
                continue
            epoch += 1
            batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
            state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
            state_batch = paddle.stack(tuple(state for state in state_batch))
            reward_batch = paddle.to_tensor(np.array(reward_batch, dtype=np.float32)[:, None])
            next_state_batch = paddle.stack(tuple(state for state in next_state_batch))


            q_values = model(state_batch)
            model.eval()
            with paddle.no_grad():
                next_prediction_batch = model(next_state_batch)

            model.train()
            y_batch = paddle.concat(
                tuple(reward if done else reward + opt.gamma * prediction for reward, done, prediction in
                      zip(reward_batch, done_batch, next_prediction_batch)))[:, None]


            optimizer.clear_grad()
            loss = criterion(q_values, y_batch)
            loss.backward()
            optimizer.step()

            print("Epoch: {}/{}, Action: {}, Score: {}, Tetrominoes {}, Cleared lines: {}".format(
                epoch,
                opt.num_epochs,
                action,
                final_score,
                final_tetrominoes,
                final_cleared_lines))

            writer.add_scalar(tag="Train/Score", step=epoch-1, value=final_score)
            writer.add_scalar(tag="Train/Tetrominoes", step=epoch-1, value=final_tetrominoes)
            writer.add_scalar(tag="Train/Cleared lines", step=epoch-1, value=final_cleared_lines)

            if epoch > 0 and epoch % opt.save_interval == 0:
                paddle.save(model.state_dict(), "{}/tetris_{}".format(opt.saved_path, epoch))

    paddle.save(model.state_dict(), "{}/tetris".format(opt.saved_path))


if __name__ == "__main__":
    opt = get_args()
    train(opt)
