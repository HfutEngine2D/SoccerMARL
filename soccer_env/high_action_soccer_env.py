import os, subprocess, time, signal
import numpy as np
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

import socket
from contextlib import closing

try:
    import hfo_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you can install HFO dependencies with 'pip install gym[soccer].')".format(e))

import logging
logger = logging.getLogger(__name__)

def find_free_port():
    """Find a random free port. Does not guarantee that the port will still be free after return.
    Note: HFO takes three consecutive port numbers, this only checks one.

    Source: https://github.com/crowdAI/marLo/blob/master/marlo/utils.py

    :rtype:  `int`
    """

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


class HighActionSoccerEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human']}

    def __init__(self, config):
        self.viewer = None
        self.server_process = None
        self.server_port = None
        self.hfo_path = hfo_py.get_hfo_path()
        #print(self.hfo_path)
        self._configure_environment(config)
        self.env = hfo_py.HFOEnvironment()
        if  "feature_set" in config :
            self.env.connectToServer( feature_set=config['feature_set'], config_dir=hfo_py.get_config_path(), server_port=self.server_port)
        else :
            self.env.connectToServer( config_dir=hfo_py.get_config_path(), server_port=self.server_port)
        #print("Shape =",self.env.getStateSize())
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=((self.env.getStateSize(),)), dtype=np.float32)
        # Action space omits the Tackle/Catch actions, which are useful on defense
        # 包括：
        #   Go To Ball()
        #   Move()
        #   Shoot()
        #   Dribble()
        self.action_space = spaces.Discrete(5)

        self.status = hfo_py.IN_GAME
        self._seed = -1

    def __del__(self):
        self.env.act(hfo_py.QUIT)
        self.env.step()
        os.kill(self.server_process.pid, signal.SIGINT)
        if self.viewer is not None:
            os.kill(self.viewer.pid, signal.SIGKILL)

    def _configure_environment(self, config):
        """
        Provides a chance for subclasses to override this method and supply
        a different server configuration. By default, we initialize one
        offense agent against no defenders.
        """
    
        self._start_hfo_server(**config['server_config'])

    def _start_hfo_server(self, frames_per_trial=500,
                          #untouched_time=1000, 
                          untouched_time=100, 
                          offense_agents=1,
                          defense_agents=0, offense_npcs=0,
                          defense_npcs=0, sync_mode=True, port=None,
                          offense_on_ball=0, fullstate=True, seed=-1,
                          ball_x_min=0.0, ball_x_max=0.2,
                          verbose=False, log_game=False,
                          log_dir="log"):
        """
        Starts the Half-Field-Offense server.
        frames_per_trial: Episodes end after this many steps.
        untouched_time: Episodes end if the ball is untouched for this many steps.
        offense_agents: Number of user-controlled offensive players.
        defense_agents: Number of user-controlled defenders.
        offense_npcs: Number of offensive bots.
        defense_npcs: Number of defense bots.
        sync_mode: Disabling sync mode runs server in real time (SLOW!).
        port: Port to start the server on.
        offense_on_ball: Player to give the ball to at beginning of episode.
        fullstate: Enable noise-free perception.
        seed: Seed the starting positions of the players and ball.
        ball_x_[min/max]: Initialize the ball this far downfield: [0,1]
        verbose: Verbose server messages.
        log_game: Enable game logging. Logs can be used for replay + visualization.
        log_dir: Directory to place game logs (*.rcg).
        """
        if port is None:
            port = find_free_port()
        self.server_port = port
        '''cmd = self.hfo_path + \
              " --headless --frames-per-trial %i --untouched-time %i --offense-agents %i"\
              " --defense-agents %i --offense-npcs %i --defense-npcs %i"\
              " --port %i --offense-on-ball %i --seed %i --ball-x-min %f"\
              " --ball-x-max %f --log-dir %s"\
              % (frames_per_trial, untouched_time, 
                 offense_agents,
                 defense_agents, offense_npcs, defense_npcs, port,
                 offense_on_ball, seed, ball_x_min, ball_x_max,
                 log_dir)'''
        cmd = self.hfo_path + \
              " --headless --frames-per-trial %i --offense-agents %i"\
              " --defense-agents %i --offense-npcs %i --defense-npcs %i"\
              " --port %i --offense-on-ball %i --seed %i --ball-x-min %f"\
              " --ball-x-max %f --log-dir %s"\
              % (frames_per_trial,
                 offense_agents,
                 defense_agents, offense_npcs, defense_npcs, port,
                 offense_on_ball, seed, ball_x_min, ball_x_max,
                 log_dir)
        if not sync_mode: cmd += " --no-sync"
        if fullstate:     cmd += " --fullstate"
        if verbose:       cmd += " --verbose"
        if not log_game:  cmd += " --no-logging"
        #print('Starting server with command: %s' % cmd)
        self.server_process = subprocess.Popen(cmd.split(' '), shell=False)
        time.sleep(10) # Wait for server to startup before connecting a player

    def _start_viewer(self):
        """
        Starts the SoccerWindow visualizer. Note the viewer may also be
        used with a *.rcg logfile to replay a game. See details at
        https://github.com/LARG/HFO/blob/master/doc/manual.pdf.
        """
        cmd = hfo_py.get_viewer_path() +\
              " --connect --port %d" % (self.server_port)
        self.viewer = subprocess.Popen(cmd.split(' '), shell=False)

    def step(self, action):
        self._take_action(action)
        self.status = self.env.step()
        reward = self._get_reward()
        ob = self.env.getState(action)
        episode_over = self.status != hfo_py.IN_GAME
        return ob, reward, episode_over, {'status': self.status}

    def _take_action(self, action):
        """ Converts the action space into an HFO action. """
        action_type = ACTION_LOOKUP[action]
        self.env.act(action_type)

    # def _get_reward(self,action):
    #     """ Reward is given for scoring a goal. """
    #     if self.env.playerOnBall().unum == self.unum and ACTION_LOOKUP[action] == hfo_py.AUTOPASS:
    #         return 4
    #     elif self.status == hfo_py.GOAL
    #         return 2
    #     else:
    #         return 0
    def _get_reward(self):
        """
        Agent is rewarded for minimizing the distance between itself and
        the ball, minimizing the distance between the ball and the goal,
        and scoring a goal.
        """
        current_state = self.env.getState()
        #print("State =",current_state)
        #print("len State =",len(current_state))
        ball_proximity = current_state[53]
        goal_proximity = current_state[15]
        ball_dist = 1.0 - ball_proximity
        goal_dist = 1.0 - goal_proximity
        kickable = current_state[12]
        ball_ang_sin_rad = current_state[51]
        ball_ang_cos_rad = current_state[52]
        ball_ang_rad = math.acos(ball_ang_cos_rad)
        if ball_ang_sin_rad < 0:
            ball_ang_rad *= -1.
        goal_ang_sin_rad = current_state[13]
        goal_ang_cos_rad = current_state[14]
        goal_ang_rad = math.acos(goal_ang_cos_rad)
        if goal_ang_sin_rad < 0:
            goal_ang_rad *= -1.
        alpha = max(ball_ang_rad, goal_ang_rad) - min(ball_ang_rad,
                                                      goal_ang_rad)
        ball_dist_goal = math.sqrt(ball_dist * ball_dist +
                                   goal_dist * goal_dist - 2. * ball_dist *
                                   goal_dist * math.cos(alpha))
        # Compute the difference in ball proximity from the last step
        if not self.first_step:
            ball_prox_delta = ball_proximity - self.old_ball_prox
            kickable_delta = kickable - self.old_kickable
            ball_dist_goal_delta = ball_dist_goal - self.old_ball_dist_goal
        self.old_ball_prox = ball_proximity
        self.old_kickable = kickable
        self.old_ball_dist_goal = ball_dist_goal
        reward = 0
        if not self.first_step:
            mtb = self.__move_to_ball_reward(kickable_delta, ball_prox_delta)
            ktg = 3. * self.__kick_to_goal_reward(ball_dist_goal_delta)
            eot = self.__EOT_reward()
            reward = mtb + ktg + eot
            #print("mtb: %.06f ktg: %.06f eot: %.06f"%(mtb,ktg,eot))

        self.first_step = False
        #print("r =",reward)
        return reward

    def __move_to_ball_reward(self, kickable_delta, ball_prox_delta):
        reward = 0.
        if self.env.playerOnBall().unum < 0 or self.env.playerOnBall(
        ).unum == self.unum:
            reward += ball_prox_delta
        if kickable_delta >= 1 and not self.got_kickable_reward:
            reward += 1.
            self.got_kickable_reward = True
        return reward

     
    def reset(self):
        """ Repeats NO-OP action until a new episode begins. """
        while self.status == hfo_py.IN_GAME:
            self.env.act(hfo_py.NOOP)
            self.status = self.env.step()
        while self.status != hfo_py.IN_GAME:
            self.env.act(hfo_py.NOOP)
            self.status = self.env.step()
            # prevent infinite output when server dies
            if self.status == hfo_py.SERVER_DOWN:
                raise ServerDownException("HFO server down!")
        return self.env.getState()

    def _render(self, mode='human', close=False):
        """ Viewer only supports human mode currently. """
        if close:
            if self.viewer is not None:
                os.kill(self.viewer.pid, signal.SIGKILL)
        else:
            if self.viewer is None:
                self._start_viewer()
 
    def close(self):
        if self.server_process is not None:
            try:
                os.kill(self.server_process.pid, signal.SIGKILL)
            except Exception:
                pass


class ServerDownException(Exception):
    """
    Custom error so agents can catch it and exit cleanly if the server dies unexpectedly.
    """
    pass
  

ACTION_LOOKUP = {
    0 : hfo_py.GO_TO_BALL,
    1 : hfo_py.MOVE,
    2 : hfo_py.SHOOT,
    3 : hfo_py.DRIBBLE, # Used on defense to slide tackle the ball
    4 : hfo_py.CATCH,  # Used only by goalie to catch the ball
    5 : hfo_py.AUTOPASS
}

STATUS_LOOKUP = {
    hfo_py.IN_GAME: 'IN_GAME',
    hfo_py.SERVER_DOWN: 'SERVER_DOWN',
    hfo_py.GOAL: 'GOAL',
    hfo_py.OUT_OF_BOUNDS: 'OUT_OF_BOUNDS',
    hfo_py.OUT_OF_TIME: 'OUT_OF_TIME',
    hfo_py.CAPTURED_BY_DEFENSE: 'CAPTURED_BY_DEFENSE',
}
