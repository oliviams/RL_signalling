import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from itertools import product

from game_theory_implementation.lr_dr import *
from game_theory_implementation.param import *

# S = 3  # survival probability of donor if gives resource
# V = 2  # survival probability of beneficiary if in need of resource but not given
# t = 0.2  # cost of signalling
# p = 0.7  # probability beneficiary needs resource
# r = 0.4  # coefficient of relatedness

np.random.seed(42) # to ensure stability between runs - has to be reset every time


def action_colours(val):
    color = '#8BC34A' if 'C' in str(val) else '#F44336'
    return f'background-color: {color}'


class Player():
    """Player class - currently empty class but
    may need to add functionality common to all player classes"""

    def __init__(self, strategy):
        self.strategy = strategy

    def play_turn(self, actions, op_actions):
        pass


class SimultaneousGame():
    """Simultaneous Game environment"""
    PAYOFF_1 = np.array([[V*(1-t), 1-t], [V, 1]])
    PAYOFF_2 = np.array([[1, S], [1, S]])
#     PAYOFF_1 = np.array([[1, 5], [V, 5]])
#     PAYOFF_2 = np.array([[5, S], [5, S]])

    def __init__(self, player, opponent, rounds=10):
        #         self.history = History()
        self.rounds = rounds
        self.player = player
        self.opponent = opponent
        self.ACTIONS = []
        self.OP_ACTIONS = []
        self.ACTION_PAIRS = []

        for i in range(rounds):
            # self.SimultaneousGame.append(player.play_turn(self.ACTIONS, self.OP_ACTIONS),
            #                          opponent.play_turn(self.OP_ACTIONS, self.ACTIONS))

            action = player.play_turn(self.ACTIONS, self.OP_ACTIONS)
            op_action = opponent.play_turn(self.OP_ACTIONS, self.ACTIONS)
            self.ACTIONS.append(action)
            self.OP_ACTIONS.append(op_action)
            self.ACTION_PAIRS.append((action, op_action))

    def play(self):

        return self.ACTION_PAIRS

    def result(self):  # Should this and cumulative result go in history class?
        """Score for each round (not cumulative)"""

        game = self.ACTION_PAIRS
        result = []

        for turn in game:
            if turn[0] == 'C' and turn[1] == 'C':
                result.append((SimultaneousGame.PAYOFF_1[0, 0], SimultaneousGame.PAYOFF_2[0, 0]))
            elif turn[0] == 'C' and turn[1] == 'D':
                result.append((SimultaneousGame.PAYOFF_1[0, 1], SimultaneousGame.PAYOFF_2[0, 1]))
            elif turn[0] == 'D' and turn[1] == 'C':
                result.append((SimultaneousGame.PAYOFF_1[1, 0], SimultaneousGame.PAYOFF_2[1, 0]))
            elif turn[0] == 'D' and turn[1] == 'D':
                result.append((SimultaneousGame.PAYOFF_1[1, 1], SimultaneousGame.PAYOFF_2[1, 1]))

        return result

    def cumulative_result(self):
        """Cumulative score for the game"""

        result = self.result()
        player_scores = [i[0] for i in result]
        opponent_scores = [i[1] for i in result]
        cum_result = []
        player_cum = np.round(np.cumsum(player_scores), decimals=1).tolist()
        opponent_cum = np.round(np.cumsum(opponent_scores), decimals=1).tolist()
        for p, o in zip(player_cum, opponent_cum):
            cum_result.append((p, o))

        return cum_result

    def cumulative_graph(self):
        """Graph of cumulative scores"""

#         game = self.ACTION_PAIRS

        plt.xlabel("Round")
        plt.ylabel("Score")
        plt.title("Cumulative scores")
        x = [i + 1 for i in list(range(len(self.cumulative_result())))]
        labels = [type(self.player).__name__, type(self.opponent).__name__]

        for i, lab in zip(range(2), labels):
            plt.plot(x, [pt[i] for pt in self.cumulative_result()], label=lab)
        plt.legend()
        plt.show()

    def cooperation_distribution(self):
        """Graph of the distribution of
        cooperation throughout the game"""

        game = self.ACTION_PAIRS

        player_coop = [1 if i[0] == 'C' else 0 for i in game]
        opponent_coop = [1 if i[1] == 'C' else 0 for i in game]
        sum_coop = [sum(x) for x in zip(player_coop, opponent_coop)]

        player_coop_sum = np.cumsum(player_coop)
        opponent_coop_sum = np.cumsum(opponent_coop)
        sum_coop_sum = np.cumsum(sum_coop)

        player_coop_prop = [a / b for a, b in zip(player_coop_sum, sum_coop_sum)]
        opponent_coop_prop = [a / b for a, b in zip(opponent_coop_sum, sum_coop_sum)]

        plt.stackplot(range(1, len(sum_coop) + 1), player_coop_prop, opponent_coop_prop,
                      labels=[type(self.player).__name__, type(self.opponent).__name__])
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel("Round")
        plt.ylabel("Distribution")
        plt.legend(loc='upper left')
        plt.show()

    def q_values(self):
        """Q-value graphs for each state, only for Q-learning players"""

        try:  # if only player 1 is q-learner, this will fail when it gets to the opponent, otherwise will create graphs for both
            player_states = list(self.player.state_dict.keys())
            player_qs = self.player.q_values
            for i, s in zip(range(0, len(player_qs), 2), player_states):
                plt.plot(range(0, self.rounds), player_qs[i], label='Action: C')
                plt.plot(range(0, self.rounds), player_qs[i + 1], label='Action: D')
                plt.ylabel('State: ' + s)
                plt.legend()
                plt.show()

            opponent_states = list(self.opponent.state_dict.keys())
            opponent_qs = self.opponent.q_values
            for i, s in zip(range(0, len(opponent_qs), 2), opponent_states):
                plt.plot(range(0, self.rounds), opponent_qs[i], label='Action: C')
                plt.plot(range(0, self.rounds), opponent_qs[i + 1], label='Action: D')
                plt.ylabel('State: ' + s)
                plt.legend()
                plt.show()

        except:
            try:  # if only opponent is q-learner
                opponent_states = list(self.opponent.state_dict.keys())
                opponent_qs = self.opponent.q_values
                for i, s in zip(range(0, len(opponent_qs), 2), opponent_states):
                    plt.plot(range(0, self.rounds), opponent_qs[i], label='Action: C')
                    plt.plot(range(0, self.rounds), opponent_qs[i + 1], label='Action: D')
                    plt.ylabel('State: ' + s)
                    plt.legend()
                    plt.show()

            except:
                pass

    def last_50(self):
        """Last 50 moves, C in green and D in red"""

        df = pd.DataFrame({'Player 1': self.ACTIONS[-50:], 'Player 2': [i.upper() for i in self.OP_ACTIONS[-50:]]})
        return df.style.applymap(action_colours)


class SequentialGame(SimultaneousGame):
    """Sequential Game environment"""
    PAYOFF_1 = np.array([[V*(1-t), 1-t], [V, 1]])
    PAYOFF_2 = np.array([[1, S], [1, S]])
#     PAYOFF_1 = np.array([[1, 5], [V, 5]])
#     PAYOFF_2 = np.array([[5, S], [5, S]])

    def __init__(self, player, opponent, rounds=10):
        #         self.history = History()
        self.rounds = rounds
        self.player = player
        self.opponent = opponent
        self.ACTIONS = []
        self.OP_ACTIONS = []
        self.ACTION_PAIRS = []

        for i in range(rounds):
            self.ACTIONS.append(player.play_turn(self.ACTIONS, self.OP_ACTIONS))
            self.OP_ACTIONS.append(opponent.play_turn(self.OP_ACTIONS, self.ACTIONS).lower())

        for p, o in zip(self.ACTIONS, self.OP_ACTIONS):
            self.ACTION_PAIRS.append(p)
            self.ACTION_PAIRS.append(o)

    def result(self):  # Should this and cumulative result go in history class?
        """Score for each round (not cumulative)"""

        game = [t.upper() for t in self.ACTION_PAIRS]
        result = []

        for i in range(0, len(game), 2):
            if game[i] == 'C' and game[i + 1] == 'C':
                result.append((SequentialGame.PAYOFF_1[0, 0], SequentialGame.PAYOFF_2[0, 0]))
            elif game[i] == 'C' and game[i + 1] == 'D':
                result.append((SequentialGame.PAYOFF_1[0, 1], SequentialGame.PAYOFF_2[0, 1]))
            elif game[i] == 'D' and game[i + 1] == 'C':
                result.append((SequentialGame.PAYOFF_1[1, 0], SequentialGame.PAYOFF_2[1, 0]))
            elif game[i] == 'D' and game[i + 1] == 'D':
                result.append((SequentialGame.PAYOFF_1[1, 1], SequentialGame.PAYOFF_2[1, 1]))

        return result

    def cooperation_distribution(self):
        """Graph of the distribution of
        cooperation throughout the game"""

        game = [t.upper() for t in self.ACTION_PAIRS]

        player_coop = []
        opponent_coop = []
        for i in range(0, len(game), 2):
            if game[i] == 'C':
                player_coop.append(1)
            else:
                player_coop.append(0)

            if game[i + 1] == 'C':
                opponent_coop.append(1)
            else:
                opponent_coop.append(0)

        sum_coop = [sum(x) for x in zip(player_coop, opponent_coop)]

        player_coop_sum = np.cumsum(player_coop)
        opponent_coop_sum = np.cumsum(opponent_coop)
        sum_coop_sum = np.cumsum(sum_coop)

        player_coop_prop = [a / b for a, b in zip(player_coop_sum, sum_coop_sum)]
        opponent_coop_prop = [a / b for a, b in zip(opponent_coop_sum, sum_coop_sum)]

        plt.stackplot(range(1, len(sum_coop) + 1), player_coop_prop, opponent_coop_prop,
                      labels=[type(self.player).__name__, type(self.opponent).__name__])
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel("Round")
        plt.ylabel("Distribution")
        plt.legend(loc='upper left')
        plt.show()


class GameSwitch(SequentialGame):
    """Players switch role every turn -
    relevant for asymmetric payoff games"""

    def result(self):
        """Score for each round (not cumulative)"""

        game = [t.upper() for t in self.ACTION_PAIRS]
        result = []

        for i in range(0, len(game), 4):
            if game[i] == 'C' and game[i + 1] == 'C':
                result.append((SequentialGame.PAYOFF_1[0, 0], SequentialGame.PAYOFF_2[0, 0]))
            elif game[i] == 'C' and game[i + 1] == 'D':
                result.append((SequentialGame.PAYOFF_1[0, 1], SequentialGame.PAYOFF_2[0, 1]))
            elif game[i] == 'D' and game[i + 1] == 'C':
                result.append((SequentialGame.PAYOFF_1[1, 0], SequentialGame.PAYOFF_2[1, 0]))
            elif game[i] == 'D' and game[i + 1] == 'D':
                result.append((SequentialGame.PAYOFF_1[1, 1], SequentialGame.PAYOFF_2[1, 1]))

            if game[i + 2] == 'C' and game[i + 3] == 'C':
                result.append((SequentialGame.PAYOFF_2[0, 0], SequentialGame.PAYOFF_1[0, 0]))
            elif game[i + 2] == 'C' and game[i + 3] == 'D':
                result.append((SequentialGame.PAYOFF_2[1, 0], SequentialGame.PAYOFF_1[1, 0]))
            elif game[i + 2] == 'D' and game[i + 3] == 'C':
                result.append((SequentialGame.PAYOFF_2[0, 1], SequentialGame.PAYOFF_1[0, 1]))
            elif game[i + 2] == 'D' and game[i + 3] == 'D':
                result.append((SequentialGame.PAYOFF_2[1, 1], SequentialGame.PAYOFF_1[1, 1]))

        return result



# Strategies

class Cooperator(Player):
    """Always plays C"""

    def __init__(self):
        pass

    def play_turn(self, actions, op_actions):
        return 'C'


class Defector(Player):
    """Always plays D"""

    def __init__(self):
        pass

    def play_turn(self, actions, op_actions):
        return 'D'


class Alternator(Player):
    """Alternates between C and D each turn, starts by playing C"""

    def __init__(self):
        pass

    def play_turn(self, actions, op_actions):
        if len(actions) == 0:
            return 'C'
        elif actions[-1].upper() == 'C':
            return 'D'
        elif actions[-1].upper() == 'D':
            return 'C'


class TitForTat(Player):
    """Matches opponent's last move, starts by playing C"""

    def __init__(self):
        pass

    def play_turn(self, actions, op_actions):
        if len(op_actions) == 0:
            return 'C'
        else:
            return op_actions[-1].upper()


class CyclerCCD(Player):
    """Cycles through moves CCD"""

    def __init__(self):
        pass

    def play_turn(self, actions, op_actions):
        if len(actions) == 0:
            return 'C'
        elif len(actions) == 1:
            return 'C'
        elif actions[-1].upper() == 'D':
            return 'C'
        elif actions[-1].upper() == 'C' and actions[-2].upper() == 'C':
            return 'D'
        elif actions[-1].upper() == 'C' and actions[-2].upper() == 'D':
            return 'C'

class CyclerDDC(Player):
    """Cycles through moves DDC"""

    def __init__(self):
        pass

    def play_turn(self, actions, op_actions):
        if len(actions) == 0:
            return 'D'
        elif len(actions) == 1:
            return 'D'
        elif actions[-1].upper() == 'C':
            return 'D'
        elif actions[-1].upper() == 'D' and actions[-2].upper() == 'D':
            return 'C'
        elif actions[-1].upper() == 'D' and actions[-2].upper() == 'C':
            return 'D'        

class Random(Player):
    """Randomly selects to play C or D"""

    def __init__(self):
        pass

    def play_turn(self, actions, op_actions):
        value = np.random.randint(2)
        if value == 0:
            return 'C'
        elif value == 1:
            return 'D'


class WinStayLoseShift(Player):  # Check if there are different versions of this, as some only defect for x rounds etc.
    """Starts by cooperating, but once opponent
    defects will always defect"""

    def __init__(self):
        pass

    def play_turn(self, actions, op_actions):
        if len(op_actions) == 0:
            return 'C'
        elif 'D' in [i.upper() for i in op_actions]:
            return 'D'
        else:
            return 'C'


class QLearner(Player):
    """Learns the best strategy through Q-learning.
    Explores (plays randomly) for the first 20 turns"""

    learning_rate = 0.8
    discount_rate = 0.5

    def __init__(self, memory=1):
        self.memory = memory
        self.q_table = np.zeros([4 ** self.memory, 2])  # q-table initialised with zeros

        # all possible C & D combinations for the states
        self.state_dict = {st: i for st, i in zip([''.join(x) for x in product('CD', repeat=2 * self.memory)],
                                                  list(range(4 ** self.memory)))}
        self.action_dict = {0: 'C', 1: 'D'}

        # list of lists, each q_values[i] is one point on the q-values table
        self.q_values = [[0] for x in range((4 ** self.memory) * 2)]

    def play_turn(self, actions, op_actions):
        if len(actions) == 0:  # play random at first
            value = np.random.randint(2)
            print(self.q_table)
            if value == 0:
                return 'C'
            elif value == 1:
                return 'D'

        elif len(actions) < self.memory:  # play random at first
            for x in range(int(4 ** self.memory)):  # need to add zeros to q-values list
                for y in range(2):
                    self.q_values[(2 * x) + y].append(self.q_table[x, y])

            value = np.random.randint(2)
            print(self.q_table)
            if value == 0:
                return 'C'
            elif value == 1:
                return 'D'


        else:
            state = []
            for i in reversed(range(self.memory)):  # state is made up of past i turns
                state.append(''.join([actions[-(i + 1)].upper(), op_actions[-(i + 1)].upper()]))
            state = ''.join(state)

            if actions[-1].upper() == 'C' and op_actions[-1].upper() == 'C':
                reward = Game.PAYOFF_1[0, 0]
            elif actions[-1].upper() == 'C' and op_actions[-1].upper() == 'D':
                reward = Game.PAYOFF_1[0, 1]
            elif actions[-1].upper() == 'D' and op_actions[-1].upper() == 'C':
                reward = Game.PAYOFF_1[1, 0]
            elif actions[-1].upper() == 'D' and op_actions[-1].upper() == 'D':
                reward = Game.PAYOFF_1[1, 1]

            if len(actions) < 20:  # explore (play random) for first 20 turns
                action = np.random.randint(2)

                # update q-table and q-value list given the state (for all other states, list just updates with previous value)
                self.q_table[self.state_dict[state], action] = self.q_table[self.state_dict[
                                                                                state], action] + QLearner.learning_rate * (
                                                                           reward +
                                                                           QLearner.discount_rate * (self.q_table[
                                                                       self.state_dict[state], np.argmax(self.q_table[
                                                                                                             self.state_dict[
                                                                                                                 state], action])]) -
                                                                           self.q_table[self.state_dict[state], action])

                for x in range(4 ** self.memory):
                    for y in range(2):
                        self.q_values[(2 * x) + y].append(self.q_table[x, y])
                print(self.q_table)
                return self.action_dict[action]


            else:
                # Choose randomly if both q-values are equal
                action = np.random.choice(
                    np.where(self.q_table[self.state_dict[state]] == self.q_table[self.state_dict[state]].max())[0])

                self.q_table[self.state_dict[state], action] = self.q_table[self.state_dict[
                                                                                state], action] + QLearner.learning_rate * (
                                                                           reward +
                                                                           QLearner.discount_rate * (self.q_table[
                                                                       self.state_dict[state], np.argmax(self.q_table[
                                                                                                             self.state_dict[
                                                                                                                 state], action])]) -
                                                                           self.q_table[self.state_dict[state], action])

                for x in range(4 ** self.memory):
                    for y in range(2):
                        self.q_values[(2 * x) + y].append(self.q_table[x, y])
                print(self.q_table)
                return self.action_dict[action]