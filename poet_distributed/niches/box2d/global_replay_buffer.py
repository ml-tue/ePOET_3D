from torchrl.replay_buffers import BaseReplayBuffer
global replay_buffer
replay_buffer = BaseReplayBuffer(
        env_nums=1,
        max_replay_buffer_size=int(1e6),
        time_limit_filter=False
    )