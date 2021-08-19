import numpy as np
import mujoco_py
# import src.my_utils as my_utils
import time
import os
import cv2
from . import hf_gen
import gym
from gym import spaces
from math import acos
from gym.utils import seeding
from collections import namedtuple
from scipy import ndimage
from opensimplex import OpenSimplex
import math

Env_config = namedtuple('Env_config', [
    'name',
    'ground_roughness',
    'pit_gap',
    'stump_width',  'stump_height', 'stump_float',
    'stair_height', 'stair_width', 'stair_steps'
])

def perlin(res, h=0.2):
    oSim = OpenSimplex(seed=int(time.time()))
    height = 200

    M = math.ceil(res * 100)
    N = math.ceil(res * 200)
    mat = np.zeros((M, N))

    scale_x = np.random.randint(30, 100)
    scale_y = np.random.randint(30, 100)
    octaves = 4 # np.random.randint(1, 5)
    persistence = np.random.rand() * 0.3 + 0.3
    lacunarity = np.random.rand() + 1.5

    for i in range(M):
        for j in range(N):
            for o in range(octaves):
                sx = scale_x * (1 / (lacunarity ** o))
                sy = scale_y * (1 / (lacunarity ** o))
                amp = persistence ** o
                mat[i][j] += oSim.noise2d(i / sx, j / sy) * amp

    wmin, wmax = mat.min(), mat.max()
    mat = (mat - wmin) / (wmax - wmin) * height

    if np.random.rand() < 0.3:
        num = np.random.randint(50, 120)
        mat = np.clip(mat, num, 200)
    if np.random.rand() < 0.3:
        num = np.random.randint(120, 200)
        mat = np.clip(mat, 0, num)

    # Walls
    mat[0, 0] = 255.
    return mat, {"height" : h}


class Quad(gym.Env):
    MODELPATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "quad.xml")
    # def __init__(self, hm_fun_list, *hm_args):
    def __init__(self):
        self.level = 1
        self.h = 0.2
        self.hm_fun_list = [perlin]
        self.hm_args = (self.level,self.h)

        # External parameters
        self.joints_rads_low = np.array([-0.2, -0.6, -0.6] * 4)
        self.joints_rads_high = np.array([0.6, 0.6, 0.6] * 4)
        self.joints_rads_diff = self.joints_rads_high - self.joints_rads_low

        self.target_vel = 0.4 # Target velocity with which we want agent to move
        self.max_steps = 2000

        self.env_config = None
        self.env_params = None
        self.env_seed = None
        self._seed()

        self.camera = False

        hm_fun = np.random.choice(self.hm_fun_list)
        hm, info = hm_fun(*self.hm_args)
        # cv2.imwrite(os.path.join(os.path.dirname(os.path.realpath(__file__)), "hm.png"), hm)
        cv2.imwrite(os.path.join(os.path.dirname(os.path.realpath(__file__)), "hm_" + str(self.h) + ".png"), hm)

        # Load simulator
        while True:
            try:
                self.model = mujoco_py.load_model_from_path(Quad.MODELPATH)
                break
            except Exception:
                pass

        # self.model = mujoco_py.load_model_from_path(Quad.MODELPATH)
        self.sim = mujoco_py.MjSim(self.model)
        self.sum_r = 0.0

        self.model.hfield_size[0][2] = info["height"]
        self.reset()

        if self.camera:
            self.cam_viewer = mujoco_py.MjRenderContextOffscreen(self.sim, 0)

        #self.observation_space = spaces.Box(low=-1, high=1, dtype=np.float32, shape=(self.obs_dim,))
        #self.action_space = spaces.Box(low=-1, high=1, dtype=np.float32, shape=(self.act_dim,))

    def seed(self, seed=None):
        return self._seed(seed)

    def _seed(self, seed=None):
        self.env_seed = seed
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_env_config(self, env_config):
        self.config = env_config

    def augment(self, params):
        self.env_params = params

    def setupcam(self):
        self.viewer = mujoco_py.MjViewer(self.sim)
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = self.model.stat.extent * .3
        self.viewer.cam.lookat[0] = -0.1
        self.viewer.cam.lookat[1] = -1
        self.viewer.cam.lookat[2] = 0.5
        self.viewer.cam.elevation = -30


    def get_state(self):
        return self.sim.get_state()


    def set_state(self, qpos, qvel=None):
        qvel = np.zeros(self.q_dim) if qvel is None else qvel
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()


    def scale_action(self, action):
        return (np.array(action) * 0.5 + 0.5) * self.joints_rads_diff + self.joints_rads_low


    def scale_joints(self, joints):
        return ((np.array(joints) - self.joints_rads_low) / self.joints_rads_diff) * 2 - 1


    def get_agent_obs(self):
        qpos = self.sim.get_state().qpos.tolist()
        qvel = self.sim.get_state().qvel.tolist()
        contacts = np.array(self.sim.data.sensordata[0:4], dtype=np.float32)
        contacts[contacts > 0.05] = 1
        contacts[contacts <= 0.05] = -1

        if self.camera:
            # On board camera input
            cam_array = self.sim.render(camera_name="frontal", width=64, height=64)
            img = cv2.cvtColor(np.flipud(cam_array), cv2.COLOR_BGR2GRAY)
            #cv2.imshow('render', img)
            #cv2.waitKey(0)

            # Rangefinder data
        #r0 = self.sim.data.get_sensor('r0')
        #r1 = self.sim.data.get_sensor('r1')

        # Joints, joint velocities, quaternion, pose velocities (xd,yd,zd,thd,phd,psid), foot contacts
        return np.concatenate((self.scale_joints(qpos[7:]), qvel[6:], qpos[3:7], qvel[:6], contacts))


    def step(self, ctrl):
        obs_p = self.get_agent_obs()
        # Clip control signal
        ctrl = np.clip(ctrl, -1, 1)

        # Control penalty
        ctrl_pen = np.square(ctrl).mean()
        # Scale control according to joint ranges
        ctrl = self.scale_action(ctrl)

        # Step the simulator
        self.sim.data.ctrl[:] = ctrl
        self.sim.forward()
        self.sim.step()
        self.step_ctr += 1

        # Get agent telemetry data
        bx, by, bz, qw, qx, qy, qz = self.sim.get_state().qpos.tolist()[:7]
        xd, yd, zd, thd, phid, psid = self.sim.get_state().qvel.tolist()[:6]
        #roll, pitch, yaw = my_utils.quat_to_rpy((qw, qx, qy, qz))

        # Reward conditions
        velocity_rew = 1. / (abs(xd - self.target_vel) + 1.) - 1. / (self.target_vel + 1.)
        obs_c = self.get_agent_obs()
        # velocity_rew = obs_c[0] - obs_p[0]

        q_yaw = 2 * acos(qw)

        r = velocity_rew * 10 - \
            np.square(q_yaw) * 0.5 - \
            np.square(ctrl_pen) * 0.01 - \
            np.square(zd) * 0.5

        self.sum_r = max(self.sum_r, r)

        # Reevaluate termination condition
        # done = self.step_ctr > self.max_steps # or abs(roll) > 0.8 or abs(pitch) > 0.8
        done = bz < 0 or self.step_ctr > self.max_steps

        return obs_c, r, done, None

    @property
    def is_healthy(self):
        state = self.state_vector()
        is_healthy = np.isfinite(state).all()
        return is_healthy

    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])

    def reset(self):
        self.reset_model()

    def reset_model(self):
        # Generate environment
        if self.sum_r > 2000:
            self.h += 0.2
            self.hm_fun_list = [perlin]
            self.hm_args = (self.level, self.h)
            hm_fun = np.random.choice(self.hm_fun_list)
            hm, info = hm_fun(*self.hm_args)
            cv2.imwrite(os.path.join(os.path.dirname(os.path.realpath(__file__)), "hm.png"), hm)
            cv2.imwrite(os.path.join(os.path.dirname(os.path.realpath(__file__)), "hm_" + str(self.h) + ".png"), hm)

            # Load simulator
            while True:
                try:
                    self.model = mujoco_py.load_model_from_path(Quad.MODELPATH)
                    break
                except Exception:
                    pass

            # self.model = mujoco_py.load_model_from_path(Quad.MODELPATH)
            self.sim = mujoco_py.MjSim(self.model)

            # Set appropriate height according to height map
            self.model.hfield_size[0][2] = info["height"]
            self.sum_r = 0.0

        self.model.opt.timestep = 0.02
        self.viewer = None

        # Height field
        self.hf_data = self.model.hfield_data
        self.hf_ncol = self.model.hfield_ncol[0]
        self.hf_nrow = self.model.hfield_nrow[0]
        self.hf_column_meters = self.model.hfield_size[0][0] * 2
        self.hf_row_meters = self.model.hfield_size[0][1] * 2
        self.hf_height_meters = self.model.hfield_size[0][2]
        self.pixels_per_column = self.hf_ncol / float(self.hf_column_meters)
        self.pixels_per_row = self.hf_nrow / float(self.hf_row_meters)
        self.hf_grid = self.hf_data.reshape((self.hf_nrow, self.hf_ncol))

        local_grid = self.hf_grid[45:55, 5:15]
        max_height = np.max(local_grid) * self.hf_height_meters

        # Environment dimensions
        self.q_dim = self.sim.get_state().qpos.shape[0]
        self.qvel_dim = self.sim.get_state().qvel.shape[0]

        self.obs_dim = 12 + 12 + 4 + 6 + 4 # j, jd, quat, pose_velocity, contacts
        self.act_dim = self.sim.data.actuator_length.shape[0]

        # Set initial position
        init_q = np.zeros(self.q_dim, dtype=np.float32)
        init_q[0] = 0.0
        init_q[1] = 0.0
        init_q[2] = max_height + 0.05
        init_qvel = np.random.randn(self.qvel_dim).astype(np.float32) * 0.1

        # Set environment state
        self.set_state(init_q, init_qvel)
        self.step_ctr = 0

        obs = self.get_agent_obs()

        # obs, _, _, _ = self.step(np.zeros(self.act_dim))

        return obs

    def render(self, camera=None):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
            # self.setupcam()
        self.viewer.render()

    #
    # def demo(self):
    #     for i in range(100000):
    #         self.sim.forward()
    #         self.sim.step()
    #         self.render()

#
# if __name__ == "__main__":
#     import argparse
#
#     parser = argparse.ArgumentParser(description='Enter input arguments')
#     parser.add_argument("--terrain", type=str,
#                         help="Terrain type, choose out of: perlin,"
#                              " flat, corridor, corridor_holes, tiles,"
#                              " triangles, domes, stairs, pipe, slant, corridor_various_width, "
#                              "pipe_variable_rad, corridor_turns, pillars_random, pillars_pseudorandom")
#     args = parser.parse_args()
#
#     print("CHOSEN : {}".format(args.terrain))
#
#     terrains = {'perlin': [hf_gen.perlin],
#                 'flat': [hf_gen.flat],
#                 'corridor': [hf_gen.corridor],
#                 'corridor_holes': [hf_gen.corridor_holes],
#                 'tiles': [hf_gen.tiles],
#                 'triangles': [hf_gen.triangles],
#                 'domes': [hf_gen.domes],
#                 'stairs': [hf_gen.stairs],
#                 'pipe': [hf_gen.pipe],
#                 'slant': [hf_gen.slant],
#                 'corridor_various_width': [hf_gen.corridor_various_width],
#                 'pipe_variable_rad': [hf_gen.pipe_variable_rad],
#                 'corridor_turns': [hf_gen.corridor_turns],
#                 'pillars_random': [hf_gen.pillars_random],
#                 'pillars_pseudorandom': [hf_gen.pillars_pseudorandom]}
#
#     t = terrains[args.terrain]
#
#     quad = Quad(t, 1)
#     quad.demo()