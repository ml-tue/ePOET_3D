from . import containers
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from collections import namedtuple
from scipy import ndimage
import os
from . import rewards, enums

STEP = 0.1
TERRAIN_CMAP = 'Greens'

Env_config = namedtuple('Env_config', [
    'name',
    'ground_roughness',
    'pit_gap',
    'stump_width',  'stump_height', 'stump_float',
    'stair_height', 'stair_width', 'stair_steps'
])

DEFAULT_CAMERA_CONFIG = {
    'distance': 4.0,
}

FPS = 50
SCALE = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

VIEWPORT_W = 600
VIEWPORT_H = 400

TERRAIN_STEP = 14 / SCALE
TERRAIN_LENGTH = 100000     # in steps
TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4
# TERRAIN_HEIGHT = 1.0
TERRAIN_GRASS = 10    # low long are grass spots, in steps
TERRAIN_STARTPAD = 20    # in steps

_HEIGHTFIELD_ID = 0
_TERRAIN_SMOOTHNESS = 0.15  # 0.0: maximally bumpy; 1.0: completely smooth.
_TERRAIN_BUMP_SCALE = 2  # Spatial scale of terrain bumps (in meters).
_DEFAULT_VALUE_AT_MARGIN = 0.1

_TOES = ['toe_front_left', 'toe_back_left', 'toe_back_right', 'toe_front_right']
_WALLS = ['wall_px', 'wall_py', 'wall_nx', 'wall_ny']

DEFAULT_PATH = '/tmp/mujoco_terrains'

_SUITE_DIR = os.path.dirname(os.path.dirname(__file__))
_FILENAMES = [
    "box2d/common/materials.xml",
    "box2d/common/skybox.xml",
    "box2d/common/visual.xml",
]

_DEFAULT_TIME_LIMIT = 20
_CONTROL_TIMESTEP = .02

class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='ant_q.xml',
                 ctrl_cost_weight=0.5,
                 contact_cost_weight=5e-4,
                 healthy_reward=1.0,
                 terminate_when_unhealthy=False,
                 healthy_z_range=(0.2, 2.0),
                 contact_force_range=(-1.0, 1.0),
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True):
        utils.EzPickle.__init__(**locals())
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale

        self.env_config = None
        self.env_params = None
        self._sensor_types_to_names = {}
        self._hinge_names = []
        self._reset_next_step = True
        self._step_count = 0
        self._n_sub_steps = 0
        self._step_limit = _DEFAULT_TIME_LIMIT

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

        self.gen_terrains()

    @property
    def healthy_reward(self):
        return float(
            self.is_healthy
            or self._terminate_when_unhealthy
        ) * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.sim.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces))
        return contact_cost

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = (np.isfinite(state).all() and min_z <= state[2] <= max_z)
        return is_healthy

    @property
    def done(self):
        done = (not self.is_healthy
                if self._terminate_when_unhealthy
                else False)
        return done

    def set_env_config(self, env_config):
        self.config = env_config

    def augment(self, params):
        self.env_params = params

    def _reset_model(self):
        """Starts a new episode and returns the first `TimeStep`."""
        self._reset_next_step = False
        self._step_count = 0
        observation = self.get_observation()
        return observation

    def reset_model(self):
        """Starts a new episode and returns the first `TimeStep`."""
        self._reset_next_step = False
        self._step_count = 0
        self.gen_terrains()
        observation = self.get_observation()
        return observation

    def get_termination(self):
        """Terminates when the state norm is smaller than epsilon."""
        self.state = np.concatenate([self.data.qpos, self.data.qvel, self.data.act])
        if np.linalg.norm(self.state) < 1e-6:
            return 0.0

    def step(self, action):
        """Updates the environment using the action and returns a `TimeStep`."""
        self.do_simulation(action, self.frame_skip)

        reward = np.sum(self.get_reward())
        observation = self.get_observation()
        self._step_count += 1
        if self._step_count >= self._step_limit:
            discount = 1.0
        else:
            discount = self.get_termination()

        episode_over = discount is not None
        done = episode_over or self.done
        finish = False
        if self.sim.data.qpos[0] > (TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP:
            done = True
            finish = True

        info = {
            'finish': finish,
        }

        return observation, reward, done, info

    def _get_sensor_names(self, *sensor_types):
        try:
          sensor_names = self._sensor_types_to_names[sensor_types]
        except KeyError:
          [sensor_ids] = np.where(np.in1d(self.model.sensor_type, sensor_types))
          sensor_names = [self.model.sensor_id2name(s_id) for s_id in sensor_ids]
          self._sensor_types_to_names[sensor_types] = sensor_names
        return sensor_names

    def torso_upright(self):
        """Returns the dot-product of the torso z-axis and the global z-axis."""
        idx = self.model.body_names.index('torso')
        return np.asarray(self.data.body_xmat[idx])

    def torso_velocity(self):
        """Returns the velocity of the torso, in the local frame."""
        idx = self.model.sensor_names.index('velocimeter')
        return self.data.sensordata[idx].copy()

    def egocentric_state(self):
        """Returns the state without global orientation or position."""
        if not self._hinge_names:
            [hinge_ids] = np.nonzero(self.model.jnt_type ==
                                     enums.mjtJoint.mjJNT_HINGE)
        return [np.hstack((self.data.qpos[id],
                          self.data.qvel[id],
                          self.data.act)) for id in hinge_ids]

    def toe_positions(self):
        """Returns toe positions in egocentric frame."""
        torso_frame = self.data.xmat['torso'].reshape(3, 3)
        torso_pos = self.data.xpos['torso']
        torso_to_toe = self.data.xpos[_TOES] - torso_pos
        return torso_to_toe.dot(torso_frame)

    def force_torque(self):
        """Returns scaled force/torque sensor readings at the toes."""
        force_torque_sensors = self._get_sensor_names(enums.mjtSensor.mjSENS_FORCE,
                                                      enums.mjtSensor.mjSENS_TORQUE)
        sensor_data = []
        for name in force_torque_sensors:
            id = self.model.sensor_name2id(name)
            sensor_data.append(self.data.sensordata[id])
        return sensor_data

    def imu(self):
        """Returns IMU-like sensor readings."""
        imu_sensors = self._get_sensor_names(enums.mjtSensor.mjSENS_GYRO,
                                             enums.mjtSensor.mjSENS_ACCELEROMETER)
        sensor_data = []
        for name in imu_sensors:
            id = self.model.sensor_name2id(name)
            sensor_data.append(self.data.sensordata[id])
        return sensor_data

    def rangefinder(self):
        """Returns scaled rangefinder sensor readings."""
        rf_sensors = self._get_sensor_names(enums.mjtSensor.mjSENS_RANGEFINDER)
        sensor_data = []
        for name in rf_sensors:
            id = self.model.sensor_name2id(name)
            sensor_data.append(self.data.sensordata[id])
        rf_readings = np.array(sensor_data)
        no_intersection = -1.0
        return np.where(rf_readings == no_intersection, 1.0, np.tanh(rf_readings))

    def origin_distance(self):
        """Returns the distance from the origin to the workspace."""
        id = self.model.site_name2id('workspace')
        return np.asarray(np.linalg.norm(self.data.site_xpos[id]))

    def origin(self):
        """Returns origin position in the torso frame."""
        idx = self.model.body_names.index('torso')
        torso_frame = self.data.body_xmat[idx].reshape((3, 3))
        torso_pos = self.data.body_xpos[idx]
        return -torso_pos.dot(torso_frame)

    def target_position(self):
        """Returns target position in torso frame."""
        torso_frame = self.data.xmat['torso'].reshape(3, 3)
        torso_pos = self.data.xpos['torso']
        torso_to_target = self.data.site_xpos['target'] - torso_pos
        return torso_to_target.dot(torso_frame)

    def _find_non_contacting_height(self, orientation, x_pos=0.0, y_pos=0.0):
      z_pos = 0.0  # Start embedded in the floor.
      num_contacts = 1
      num_attempts = 0
      # Move up in 1cm increments until no contacts.
      while num_contacts > 0:
        try:
          self._reset_model()
          self.data.qpos[:3] = x_pos, y_pos, z_pos
          self.data.qpos[3:7] = orientation
        except NotImplementedError:
          pass
        num_contacts = self.data.ncon
        z_pos += 0.01
        num_attempts += 1
        if num_attempts > 10000:
          raise RuntimeError('Failed to find a non-contacting configuration.')


    def _upright_reward(self, deviation_angle=0):
      deviation = np.cos(np.deg2rad(deviation_angle))
      return rewards.tolerance(
          self.torso_upright(),
          bounds=(deviation, float('inf')),
          sigmoid='linear',
          margin=1 + deviation,
          value_at_margin=0)

    def gen_terrains(self):
        res = self.model.hfield_nrow[_HEIGHTFIELD_ID]
        assert res == self.model.hfield_ncol[_HEIGHTFIELD_ID]
        # Sinusoidal bowl shape.
        row_grid, col_grid = np.ogrid[-1:1:res*1j, -1:1:res*1j]
        radius = np.clip(np.sqrt(col_grid**2 + row_grid**2), .04, 1)
        bowl_shape = .5 - np.cos(2*np.pi*radius)/2
        # Random smooth bumps.
        terrain_size = 2 * self.model.hfield_size[_HEIGHTFIELD_ID, 0]
        bump_res = int(terrain_size / _TERRAIN_BUMP_SCALE)
        bumps = np.random.uniform(_TERRAIN_SMOOTHNESS, 1, (bump_res, bump_res))
        smooth_bumps = ndimage.zoom(bumps, res / float(bump_res))
        # Terrain is elementwise product.
        terrain = bowl_shape * smooth_bumps
        start_idx = self.model.hfield_adr[_HEIGHTFIELD_ID]
        self.model.hfield_data[start_idx:start_idx+res**2] = terrain.ravel()
        # super().initialize_episode(physics)

        orientation = np.random.randn(4)
        orientation /= np.linalg.norm(orientation)
        self._find_non_contacting_height(orientation)


    def get_observation(self):
        return np.concatenate([
            np.array(self.egocentric_state()).reshape(-1),
            np.array(self.torso_velocity()).reshape(-1),
            np.array(self.torso_upright()).reshape(-1),
            np.array(self.imu()).reshape(-1),
            np.array(self.force_torque()).reshape(-1),
            np.array(self.origin()).reshape(-1),
            np.array(self.rangefinder()).reshape(-1),]).reshape(-1)

    def get_reward(self):
        terrain_size = self.model.hfield_size[_HEIGHTFIELD_ID, 0]
        escape_reward = rewards.tolerance(
            self.origin_distance(),
            bounds=(terrain_size, float('inf')),
            margin=terrain_size,
            value_at_margin=0,
            sigmoid='linear')
        return self._upright_reward(deviation_angle=20) * escape_reward