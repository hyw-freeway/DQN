import argparse
import paddle
import cv2
from src.tetris import Tetris
from src.deep_q_network import DeepQNetwork
from paddle.distributed import fleet
from paddle.io import DataLoader, DistributedBatchSampler

paddle.set_device('gpu')

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")

    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--fps", type=int, default=300, help="frames per second")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--output", type=str, default="output.mp4")

    args = parser.parse_args()
    return args


def test(opt):
    
    model = DeepQNetwork()
    load_layer_state_dict = paddle.load("{}/tetris".format(opt.saved_path))
    model.set_dict(load_layer_state_dict)
    model.eval()
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    env.reset()

    out = cv2.VideoWriter(opt.output, cv2.VideoWriter_fourcc(*"MJPG"), opt.fps,
                          (int(1.5 * opt.width * opt.block_size), opt.height * opt.block_size))
    while True:
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = paddle.stack(next_states)

        predictions = model(next_states)
        predictions = predictions[:, 0]
        index = paddle.argmax(predictions).item()
        action = next_actions[index]
        _, done = env.step(action, render=True, video=out)

        if done:
            out.release()
            break


if __name__ == "__main__":
    opt = get_args()
    test(opt)
