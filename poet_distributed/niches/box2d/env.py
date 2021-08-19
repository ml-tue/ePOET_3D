# The following code is modified from hardmaru/estool (https://github.com/hardmaru/estool/) under the MIT License.

# Modifications Copyright (c) 2020 Uber Technologies, Inc.


from collections import namedtuple
# # from .ant_custom import AntEnv, Env_config
# # from .hill_ant import AntHillEnv
# # from .ant_q import AntEnv
# from .ant_custom import Env_config
# # from .ant_n import AntEnv
# from .ant_v3 import AntEnv
# from .ant_terrain_mjc.ant_terrain_mjc import AntTerrainMjc
# # from .quad_locomotion import hf_gen
# from .quad_locomotion.quad_blind import Quad, Env_config
# from .ant_terrain_mjc.ant_blind import AntTerrain
# # from .ant_terrain_mjc.hex_blind import Hexapod
from .ant_terrain_mjc.hexapod_trossen_terrain_all import Hexapod, Env_config

def make_env(env_name, seed, render_mode=False, env_config=None):
    if env_name.startswith("Hexapod"):
        assert env_config is not None
        # env = AntEnv(xml_file='/home/TUE/20191160/ant_poet/poet/poet_distributed/niches/box2d/ant.xml')
        # env = AntEnv(xml_file='/home/fang/project/thesis/ant_poet/poet/poet_distributed/niches/box2d/ant.xml')
        # env = AntTerrainMjc()
        # env = Quad()
        # env = AntTerrain()
        env = Hexapod()
        # env = AntTerrainMjc(xml_file='/home/fang/project/thesis/ant_poet/poet/poet_distributed/niches/box2d/ant_terrain_mjc/assets/ant_terrain_mjc.xml')
        # env = AntHillEnv(xml_file='/home/fang/project/thesis/ant_poet/poet/poet_distributed/niches/box2d/hill_ant.xml')
        # env = AntEnv(xml_file='/home/fang/project/thesis/ant_poet/poet/poet_distributed/niches/box2d/quadruped.xml')
    else:
        # env = gym.make(env_name)
        raise Exception('Got env_name {}'.format(env_name))
    # if render_mode and not env_name.startswith("Roboschool"):
    #     env.render("human")
    if (seed >= 0):
         env.seed(seed)
    return env


Game = namedtuple('Game', ['env_name', 'time_factor', 'input_size',
                           'output_size', 'layers', 'activation', 'noise_bias',
                           'output_noise'])

env_custom = Game(env_name='Hexapod',
                        # input_size=115,
                        # output_size=8,
                        # input_size=38,
                        # output_size=12,
                        input_size=53,
                        output_size=18,
                        time_factor=0,
                        layers=[40, 40],
                        activation='tanh',
                        noise_bias=0.0,
                        output_noise=[False, False, False],
                        )
