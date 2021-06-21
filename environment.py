import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import glob
import pickle

from math import copysign
pd.options.mode.chained_assignment = None


INITIAL_ACCOUNT_BALANCE = 100_000


class StockTradingEnv(gym.Env):
    """ A stock trading environment for OpenAI gym
        StockTradingEnv2 uses pre-processed staged data
        This is only used for the reinforcement learning stage
        Generate the staged data via stage_data.py
    """

    metadata = {'render.modes': ['human']}
    ACTION_SPACE_SIZE = 3 # Buy, Sell or Hold


    def __init__(self, close, num_steps):
        super(StockTradingEnv, self).__init__()

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6, 6), dtype=np.float16)

        # Data
        self.max_steps = 0
        self.current_step = 0
        self.close = close
        self.num_steps = num_steps



    def _take_action(self, action):
        # Set the current price
        self.current_price = self.close.iloc[self.current_step]
        # print('current_price: ',self.current_price)
        self.buy_n_hold = (self.current_price / self.initial_price) - 1

        comission = 0.0 # The comission is applied to both buy and sell
        amount = 1

        if action == 1:
            # Buy amount % of balance in shares
            total_possible = int(self.balance / self.current_price)
            shares_bought = int(total_possible * amount)
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * self.current_price * (1 + comission)

            self.balance -= additional_cost
            self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought
            self.shares_bought_total += additional_cost
            self.buy_triggers += 1
            

        elif action == 2:
            # Sell amount % of shares held
            self.shares_sold = int(self.shares_held * amount)
            self.balance += self.shares_sold * self.current_price * (1 - comission)
            self.shares_held -= self.shares_sold
            self.total_shares_sold += self.shares_sold
            self.total_sales_value += self.shares_sold * self.current_price
            self.sell_triggers += 1

        elif action == 0:
            self.hold_triggers += 1


        # Save amount
        self.amounts.append(amount)

        # Update the net worth
        self.net_worth = self.balance + self.shares_held * self.current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0



    def step(self, action):

        # Execute one time step within the environment
        self._take_action(action)

        # Check if there are no more steps in the data or we have met the maximum amount of steps
        if self.current_step == self.max_steps or self.current_step == len(self.close):
            done = True
        else:
            done = False

        # Calculate the reward
        reward = (self.net_worth / INITIAL_ACCOUNT_BALANCE) - 1

        # Done if net worth is negative
        if not done:
            done = self.net_worth <= 0

        # Iterate step
        self.current_step += 1

        return reward, done



    def reset(self):

        # Reset the state of the environment to an initial state
        self.episode = 1
        self.ticker = None
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.shares_bought_total = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.buy_triggers = 0
        self.sell_triggers = 0
        self.hold_triggers = 0
        self.amounts = list()
        self.buy_n_hold = 0
        

        # Set initial values
        self.current_step = random.randint(0,len(self.close)-(self.max_steps + self.num_steps))
        self.initial_price = self.close[self.current_step]
        self.last_buy_price = self.initial_price




if __name__ == '__main__':
    pass