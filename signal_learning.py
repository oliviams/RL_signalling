import numpy as np
import pandas as pd
from itertools import product
from itertools import permutations
import copy

from game_theory_implementation.ps_setup import *

def signal_colours(val):
    color = '#8BC34A' if 'C' in str(val) else '#F44336' if 'D' in str(val) else '#dceffc' if 'N' in str(val) else '#72a3d4' if 'W' in str(val) else '#3232a8' if 'S' in str(val) else '#fcfcfc'
    return f'background-color: {color}'

def rotations(li):
    count = 0
    while count < len(li):
        yield tuple(li)
        li = li[1:] + [li[0]]
        count += 1 

def unique_patterns(x, repeat):
    """Creates list of unique patterns 
    given list of actions and pattern length"""
    
    all_patterns = [p for p in product(x, repeat=repeat)]
    set_patterns = set(all_patterns)
    
    unique_patterns = []
    
    for i in all_patterns:
        if len(all_patterns) > 0:
            unique_patterns.append([item for item in all_patterns[0]])
            cycles = set(((i for i in rotations([i for i in all_patterns[0]]))))
            set_patterns = set_patterns.difference(cycles)
            all_patterns = list(set_patterns)

        else:
            pass
    
    return unique_patterns

def factorise(num):
    """Returns all factors of a number 
    excluding 1 and the number itself"""
    factors = [n for n in range(2, num) if num % n == 0]
    return factors

def remove_factors(x, length):
    """Remove subpatterns by deleting patterns
    of length of the factors"""
    
    patterns = [p for p in product(x, repeat=length)]
    factors = factorise(length)
    
    repeats = []
    for i in factors:
        pat_factors = [p for p in product(x, repeat=i)]
        list_of_lists = [list(tup) for tup in pat_factors]

        for pattern in list_of_lists:
            repeats.append(int(length/i)*pattern)
            
    if length != 2:
        for act in x:
            repeats.append(length*[act])
        
    differences = []
    for l in unique_patterns(x, length):
        if l not in repeats:
            differences.append(l)

    return sorted(differences)


signal_strengths = ['N', 'W', 'S']

# change payoffs so they can be calculated automatically using above list? List may need to be numbers for t then
# PAYOFF_1 = np.array([[1, 0], [0.75, 0], [0.25, 0]])
# PAYOFF_2 = np.array([[S, 1], [S, 1], [S, 1]])


# LEARNING_RATE = 0.1
# DISCOUNT_RATE = 0.9
# EXPLORATION = 50

# GAME

class SignalGame(SequentialGame):
    # Need to incorporate thirsty / not thirsty - start with one scenario?
    # THIRSTY
    # PAYOFF_1 = np.array([[1, 0], [0.75, 0], [0.25, 0]])
    # PAYOFF_2 = np.array([[S, 1], [S, 1], [S, 1]])

    # NOT THIRSTY
#     PAYOFF_1 = np.array([[1, V], [0.75, 0.75*V], [0.25, 0.25*V]])
#     PAYOFF_2 = np.array([[S, 1], [S, 1], [S, 1]])
    
    PAYOFF_1 = np.array([[1*r, V*r], [(1-t_w)*r, (1-t_w)*V*r], [(1-t_s)*r, (1-t_s)*V*r]])
    PAYOFF_2 = np.array([[S*r, 1*r], [S*r, 1*r], [S*r, 1*r]])

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
            self.OP_ACTIONS.append(opponent.play_turn(self.OP_ACTIONS, self.ACTIONS))

            if 'QLearner' in type(self.player).__name__:
                action_dict_rev = {'N': 0, 'W': 1, 'S': 2}

                if self.ACTIONS[-1] == 'N' and self.OP_ACTIONS[-1] == 'C':
                    reward = SignalGame.PAYOFF_1[0, 0]
                elif self.ACTIONS[-1] == 'W' and self.OP_ACTIONS[-1] == 'C':
                    reward = SignalGame.PAYOFF_1[1, 0]
                elif self.ACTIONS[-1] == 'S' and self.OP_ACTIONS[-1] == 'C':
                    reward = SignalGame.PAYOFF_1[2, 0]
                elif self.ACTIONS[-1] == 'N' and self.OP_ACTIONS[-1] == 'D':
                    reward = SignalGame.PAYOFF_1[0, 1]
                elif self.ACTIONS[-1] == 'W' and self.OP_ACTIONS[-1] == 'D':
                    reward = SignalGame.PAYOFF_1[1, 1]
                elif self.ACTIONS[-1] == 'S' and self.OP_ACTIONS[-1] == 'D':
                    reward = SignalGame.PAYOFF_1[2, 1]

                if len(self.ACTIONS) == 0:
                    pass

                if len(self.ACTIONS) < self.player.memory + 1:  # play random at first
                    for x in range(int(6 ** self.player.memory)):  # need to add zeros to q-values list
                        for y in range(3):
                            self.player.q_values[(3 * x) + y].append(self.player.q_table[x, y])

                else:
                    state = []
                    for j in reversed(range(self.player.memory)):  # state is made up of past i turns
                        state.append(''.join([self.ACTIONS[-(j + 2)], self.OP_ACTIONS[-(j + 2)]]))
                    state = ''.join(state)

                    action = action_dict_rev[self.ACTIONS[-1]]

                    # update q-table and q-value list given the state (for all other states, list just updates with previous value)
                    self.player.q_table[self.player.state_dict[state], action] = self.player.q_table[
                                                                                     self.player.state_dict[
                                                                                         state], action] + self.player.learning_rate * (
                                                                                         reward + self.player.discount_rate * (
                                                                                 self.player.q_table[
                                                                                     self.player.state_dict[
                                                                                         state], np.argmax(
                                                                                         self.player.q_table[
                                                                                             self.player.state_dict[
                                                                                                 state], action])]) -
                                                                                         self.player.q_table[
                                                                                             self.player.state_dict[
                                                                                                 state], action])

                    for x in range(6 ** self.player.memory):
                        for y in range(3):
                            self.player.q_values[(3 * x) + y].append(self.player.q_table[x, y])

            if 'QLearner' in type(self.opponent).__name__:
                action_dict_rev = {'C': 0, 'D': 1}

                if self.OP_ACTIONS[-1] == 'C' and self.ACTIONS[-1] == 'N':
                    reward = SignalGame.PAYOFF_2[0, 0]
                elif self.OP_ACTIONS[-1] == 'C' and self.ACTIONS[-1] == 'W':
                    reward = SignalGame.PAYOFF_2[1, 0]
                elif self.OP_ACTIONS[-1] == 'C' and self.ACTIONS[-1] == 'S':
                    reward = SignalGame.PAYOFF_2[2, 0]
                elif self.OP_ACTIONS[-1] == 'D' and self.ACTIONS[-1] == 'N':
                    reward = SignalGame.PAYOFF_2[0, 1]
                elif self.OP_ACTIONS[-1] == 'D' and self.ACTIONS[-1] == 'W':
                    reward = SignalGame.PAYOFF_2[1, 1]
                elif self.OP_ACTIONS[-1] == 'D' and self.ACTIONS[-1] == 'S':
                    reward = SignalGame.PAYOFF_2[2, 1]

                if len(self.OP_ACTIONS) == 0:
                    pass

                if len(self.OP_ACTIONS) < self.opponent.memory + 1:  # play random at first
                    for x in range(int(6 ** self.opponent.memory)):  # need to add zeros to q-values list
                        for y in range(2):
                            self.opponent.q_values[(2 * x) + y].append(self.opponent.q_table[x, y])

                else:
                    state = []
                    for j in reversed(range(self.opponent.memory)):  # state is made up of past i turns
                        state.append(''.join([self.OP_ACTIONS[-(j + 2)], self.ACTIONS[-(j + 2)]]))
                    state = ''.join(state)

                    action = action_dict_rev[self.OP_ACTIONS[-1]]

                    self.opponent.q_table[self.opponent.state_dict[state], action] = self.opponent.q_table[
                                                                                         self.opponent.state_dict[
                                                                                             state], action] + self.opponent.learning_rate * (
                                                                                             reward + self.opponent.discount_rate * (
                                                                                     self.opponent.q_table[
                                                                                         self.opponent.state_dict[
                                                                                             state], np.argmax(
                                                                                             self.opponent.q_table[
                                                                                                 self.opponent.state_dict[
                                                                                                     state], action])]) -
                                                                                             self.opponent.q_table[
                                                                                                 self.opponent.state_dict[
                                                                                                     state], action])

                    for x in range(6 ** self.opponent.memory):
                        for y in range(2):
                            self.opponent.q_values[(2 * x) + y].append(self.opponent.q_table[x, y])

        for p, o in zip(self.ACTIONS, self.OP_ACTIONS):
            self.ACTION_PAIRS.append(p)
            self.ACTION_PAIRS.append(o)

    def result(self):
        """Score for each round (not cumulative)"""

        game = [t for t in self.ACTION_PAIRS]
        result = []

        for i in range(0, len(game), 2):
            if game[i] == 'N' and game[i + 1] == 'C':
                result.append((SignalGame.PAYOFF_1[0, 0], SignalGame.PAYOFF_2[0, 0]))
            elif game[i] == 'W' and game[i + 1] == 'C':
                result.append((SignalGame.PAYOFF_1[1, 0], SignalGame.PAYOFF_2[1, 0]))
            elif game[i] == 'S' and game[i + 1] == 'C':
                result.append((SignalGame.PAYOFF_1[2, 0], SignalGame.PAYOFF_2[2, 0]))
            elif game[i] == 'N' and game[i + 1] == 'D':
                result.append((SignalGame.PAYOFF_1[0, 1], SignalGame.PAYOFF_2[0, 1]))
            elif game[i] == 'W' and game[i + 1] == 'D':
                result.append((SignalGame.PAYOFF_1[1, 1], SignalGame.PAYOFF_2[1, 1]))
            elif game[i] == 'S' and game[i + 1] == 'D':
                result.append((SignalGame.PAYOFF_1[2, 1], SignalGame.PAYOFF_2[2, 1]))

        return result

    def q_values(self):
        """Q-value graphs for each state, only for Q-learning players"""

        try:
            # if only player 1 is q-learner, this will fail when it gets to the opponent, otherwise will create graphs for both
            # in this case player 1 is always beneficiary and player 2 (opponent) is donor
            player_states = list(self.player.state_dict.keys())
            player_qs = self.player.q_values
            for i, s in zip(range(0, len(player_qs), 3), player_states):
                plt.plot(range(0, self.rounds + 1), player_qs[i], label='Action: N')
                plt.plot(range(0, self.rounds + 1), player_qs[i + 1], label='Action: W')
                plt.plot(range(0, self.rounds + 1), player_qs[i + 2], label='Action: S')
                plt.ylabel('State: ' + s)
                plt.legend()
                plt.show()

            opponent_states = list(self.opponent.state_dict.keys())
            opponent_qs = self.opponent.q_values
            for i, s in zip(range(0, len(opponent_qs), 2), opponent_states):
                plt.plot(range(0, self.rounds + 1), opponent_qs[i], label='Action: C')
                plt.plot(range(0, self.rounds + 1), opponent_qs[i + 1], label='Action: D')
                plt.ylabel('State: ' + s)
                plt.legend()
                plt.show()

        except:
            try:  # if only opponent is q-learner
                opponent_states = list(self.opponent.state_dict.keys())
                opponent_qs = self.opponent.q_values
                for i, s in zip(range(0, len(opponent_qs), 2), opponent_states):
                    plt.plot(range(0, self.rounds + 1), opponent_qs[i], label='Action: C')
                    plt.plot(range(0, self.rounds + 1), opponent_qs[i + 1], label='Action: D')
                    plt.ylabel('State: ' + s)
                    plt.legend()
                    plt.show()

            except:
                pass

    def q_values_list(self):
        try:
            player_qs = self.player.q_values
            opponent_qs = self.opponent.q_values
            return player_qs, opponent_qs

        except:
            try:  # if only opponent is q-learner
                player_qs = self.player.q_values
                return player_qs

            except:
                try:
                    opponent_qs = self.opponent.q_values
                    return opponent_qs

                except:
                    pass

    def last_50(self):
        """Last 50 moves, colourcoded"""

        df = pd.DataFrame({'Player 1': self.ACTIONS[-50:], 'Player 2': [i for i in self.OP_ACTIONS[-50:]]})
        return df.style.applymap(signal_colours)

    def last_x(self, x=50):
        """ Last x moves, colourcoded"""

        df = pd.DataFrame({'Player 1': self.ACTIONS[-x:], 'Player 2': [i for i in self.OP_ACTIONS[-x:]]})
        return df.style.applymap(signal_colours)
    
    def resulting_strategies(self, moves = 10):
        """Resulting strategy, considering last x moves"""
        
        player_1_acts = self.ACTIONS[-moves:]
        player_2_acts = self.OP_ACTIONS[-moves:]
        
        # Player 1
        acts_1 = ['N', 'S', 'W']
        patterns_1 = []
        for p in range(2, moves+1):
            patterns_1.append(remove_factors(acts_1, p))
        patterns_1 = [item for sublist in patterns_1 for item in sublist]    
        
        patterns_1_copy = copy.deepcopy(patterns_1)
        for i in patterns_1_copy:
            for it in range(moves):
                if len(i) < moves:
                    i += i
                else:
                    pass  
                
        dict_patterns_1 = {}
        for pat, lis in zip(patterns_1, patterns_1_copy):
            dict_patterns_1[tuple(pat)] = [list(tup)[0:moves] for tup in list(set(i for i in rotations([i for i in lis])))]  
            
        strategy_1 = [i for i in dict_patterns_1 if player_1_acts in dict_patterns_1[i]]
        
        # Player 2
        acts_2 = ['C', 'D']
        patterns_2 = []
        for p in range(2, moves+1):
            patterns_2.append(remove_factors(acts_2, p))
        patterns_2 = [item for sublist in patterns_2 for item in sublist]    
        
        patterns_2_copy = copy.deepcopy(patterns_2)
        for i in patterns_2_copy:
            for it in range(moves):
                if len(i) < moves:
                    i += i
                else:
                    pass  
                
        dict_patterns_2 = {}
        for pat, lis in zip(patterns_2, patterns_2_copy):
            dict_patterns_2[tuple(pat)] = [list(tup)[0:moves] for tup in list(set(i for i in rotations([i for i in lis])))]  
            
        strategy_2 = [i for i in dict_patterns_2 if player_2_acts in dict_patterns_2[i]]
        
        # Taking only the first (shortest) pattern, as this may return more than one strategy for the given x moves
        return (''.join(strategy_1[0]), ''.join(strategy_2[0]))
    
   

 
class PSGame(SignalGame):
    """Includes relatedness coefficient.
    Payoff is given by inclusive fitness, so 
    agent's reward + r * opponent's reward"""
    
    PAYOFF_1 = np.array([[1*r, V*r], [(1-t_w)*r, (1-t_w)*V*r], [(1-t_s)*r, (1-t_s)*V*r]])
    PAYOFF_2 = np.array([[S*r, 1*r], [S*r, 1*r], [S*r, 1*r]])

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
            self.OP_ACTIONS.append(opponent.play_turn(self.OP_ACTIONS, self.ACTIONS))

            if 'QLearner' in type(self.player).__name__:
                action_dict_rev = {'N': 0, 'W': 1, 'S': 2}
                              
                if self.ACTIONS[-1] == 'N' and self.OP_ACTIONS[-1] == 'C':
                    reward = SignalGame.PAYOFF_1[0, 0] + r * SignalGame.PAYOFF_2[0, 0]
                elif self.ACTIONS[-1] == 'W' and self.OP_ACTIONS[-1] == 'C':
                    reward = SignalGame.PAYOFF_1[1, 0] + r * SignalGame.PAYOFF_2[1, 0]
                elif self.ACTIONS[-1] == 'S' and self.OP_ACTIONS[-1] == 'C':
                    reward = SignalGame.PAYOFF_1[2, 0] + r * SignalGame.PAYOFF_2[2, 0]
                elif self.ACTIONS[-1] == 'N' and self.OP_ACTIONS[-1] == 'D':
                    reward = SignalGame.PAYOFF_1[0, 1] + r * SignalGame.PAYOFF_2[0, 1]
                elif self.ACTIONS[-1] == 'W' and self.OP_ACTIONS[-1] == 'D':
                    reward = SignalGame.PAYOFF_1[1, 1] + r * SignalGame.PAYOFF_2[1, 1]
                elif self.ACTIONS[-1] == 'S' and self.OP_ACTIONS[-1] == 'D':
                    reward = SignalGame.PAYOFF_1[2, 1] + r * SignalGame.PAYOFF_2[2, 1]

                if len(self.ACTIONS) == 0:
                    pass

                if len(self.ACTIONS) < self.player.memory + 1:  # play random at first
                    for x in range(int(6 ** self.player.memory)):  # need to add zeros to q-values list
                        for y in range(3):
                            self.player.q_values[(3 * x) + y].append(self.player.q_table[x, y])

                else:
                    state = []
                    for j in reversed(range(self.player.memory)):  # state is made up of past i turns
                        state.append(''.join([self.ACTIONS[-(j + 2)], self.OP_ACTIONS[-(j + 2)]]))
                    state = ''.join(state)

                    action = action_dict_rev[self.ACTIONS[-1]]

                    # update q-table and q-value list given the state (for all other states, list just updates with previous value)
                    self.player.q_table[self.player.state_dict[state], action] = self.player.q_table[
                                                                                     self.player.state_dict[
                                                                                         state], action] + self.player.learning_rate * (
                                                                                         reward + self.player.discount_rate * (
                                                                                 self.player.q_table[
                                                                                     self.player.state_dict[
                                                                                         state], np.argmax(
                                                                                         self.player.q_table[
                                                                                             self.player.state_dict[
                                                                                                 state], action])]) -
                                                                                         self.player.q_table[
                                                                                             self.player.state_dict[
                                                                                                 state], action])

                    for x in range(6 ** self.player.memory):
                        for y in range(3):
                            self.player.q_values[(3 * x) + y].append(self.player.q_table[x, y])

            if 'QLearner' in type(self.opponent).__name__:
                action_dict_rev = {'C': 0, 'D': 1}

                if self.OP_ACTIONS[-1] == 'C' and self.ACTIONS[-1] == 'N':
                    reward = SignalGame.PAYOFF_2[0, 0] + r * SignalGame.PAYOFF_1[0, 0]
                elif self.OP_ACTIONS[-1] == 'C' and self.ACTIONS[-1] == 'W':
                    reward = SignalGame.PAYOFF_2[1, 0] + r * SignalGame.PAYOFF_1[1, 0]
                elif self.OP_ACTIONS[-1] == 'C' and self.ACTIONS[-1] == 'S':
                    reward = SignalGame.PAYOFF_2[2, 0] + r * SignalGame.PAYOFF_1[2, 0]
                elif self.OP_ACTIONS[-1] == 'D' and self.ACTIONS[-1] == 'N':
                    reward = SignalGame.PAYOFF_2[0, 1] + r * SignalGame.PAYOFF_1[0, 1]
                elif self.OP_ACTIONS[-1] == 'D' and self.ACTIONS[-1] == 'W':
                    reward = SignalGame.PAYOFF_2[1, 1] + r * SignalGame.PAYOFF_1[1, 1]
                elif self.OP_ACTIONS[-1] == 'D' and self.ACTIONS[-1] == 'S':
                    reward = SignalGame.PAYOFF_2[2, 1] + r * SignalGame.PAYOFF_1[2, 1]

                if len(self.OP_ACTIONS) == 0:
                    pass

                if len(self.OP_ACTIONS) < self.opponent.memory + 1:  # play random at first
                    for x in range(int(6 ** self.opponent.memory)):  # need to add zeros to q-values list
                        for y in range(2):
                            self.opponent.q_values[(2 * x) + y].append(self.opponent.q_table[x, y])

                else:
                    state = []
                    for j in reversed(range(self.opponent.memory)):  # state is made up of past i turns
                        state.append(''.join([self.OP_ACTIONS[-(j + 2)], self.ACTIONS[-(j + 2)]]))
                    state = ''.join(state)

                    action = action_dict_rev[self.OP_ACTIONS[-1]]

                    self.opponent.q_table[self.opponent.state_dict[state], action] = self.opponent.q_table[
                                                                                         self.opponent.state_dict[
                                                                                             state], action] + self.opponent.learning_rate * (
                                                                                             reward + self.opponent.discount_rate * (
                                                                                     self.opponent.q_table[
                                                                                         self.opponent.state_dict[
                                                                                             state], np.argmax(
                                                                                             self.opponent.q_table[
                                                                                                 self.opponent.state_dict[
                                                                                                     state], action])]) -
                                                                                             self.opponent.q_table[
                                                                                                 self.opponent.state_dict[
                                                                                                     state], action])

                    for x in range(6 ** self.opponent.memory):
                        for y in range(2):
                            self.opponent.q_values[(2 * x) + y].append(self.opponent.q_table[x, y])

        for p, o in zip(self.ACTIONS, self.OP_ACTIONS):
            self.ACTION_PAIRS.append(p)
            self.ACTION_PAIRS.append(o)

    def result(self):
        """Score for each round (not cumulative)"""

        game = [t for t in self.ACTION_PAIRS]
        result = []

        for i in range(0, len(game), 2):
            if game[i] == 'N' and game[i + 1] == 'C':
                result.append((SignalGame.PAYOFF_1[0, 0] + r * SignalGame.PAYOFF_2[0, 0], SignalGame.PAYOFF_2[0, 0] + r * SignalGame.PAYOFF_1[0, 0]))
            elif game[i] == 'W' and game[i + 1] == 'C':
                result.append((SignalGame.PAYOFF_1[1, 0] + r * SignalGame.PAYOFF_2[1, 0], SignalGame.PAYOFF_2[1, 0] + r * SignalGame.PAYOFF_1[1, 0]))
            elif game[i] == 'S' and game[i + 1] == 'C':
                result.append((SignalGame.PAYOFF_1[2, 0] + r * SignalGame.PAYOFF_2[2, 0], SignalGame.PAYOFF_2[2, 0] + r * SignalGame.PAYOFF_1[2, 0]))
            elif game[i] == 'N' and game[i + 1] == 'D':
                result.append((SignalGame.PAYOFF_1[0, 1] + r * SignalGame.PAYOFF_2[0, 1], SignalGame.PAYOFF_2[0, 1] + r * SignalGame.PAYOFF_1[0, 1]))
            elif game[i] == 'W' and game[i + 1] == 'D':
                result.append((SignalGame.PAYOFF_1[1, 1] + r * SignalGame.PAYOFF_2[1, 1], SignalGame.PAYOFF_2[1, 1] + r * SignalGame.PAYOFF_1[1, 1]))
            elif game[i] == 'S' and game[i + 1] == 'D':
                result.append((SignalGame.PAYOFF_1[2, 1] + r * SignalGame.PAYOFF_2[2, 1], SignalGame.PAYOFF_2[2, 1] + r * SignalGame.PAYOFF_1[2, 1]))

        return result
    
    
    
    
    

# PLAYER 1

class NeverSignal(Player):
    """Never signals"""

    def __init__(self):
        pass

    def play_turn(self, actions, op_actions):
        return 'N'


class WeakSignal(Player):
    """Always signals weakly"""

    def __init__(self):
        pass

    def play_turn(self, actions, op_actions):
        return 'W'


class StrongSignal(Player):
    """Always signals strongly"""

    def __init__(self):
        pass

    def play_turn(self, actions, op_actions):
        return 'S'


class SignalQLearner(QLearner):
    """Learns the best signal strength through Q-learning.
    Explores (plays randomly) for the first x (EXPLORATION variable) turns"""

    def __init__(self, memory=1):
        self.learning_rate = LEARNING_RATE
        self.discount_rate = DISCOUNT_RATE

        self.memory = memory
        self.q_table = np.zeros([6 ** self.memory, 3])  # q-table initialised with zeros

        # all possible action combinations for the states
        self.action_combo = []
        for i in ['C', 'D']:
            for j in ['N', 'W', 'S']:
                self.action_combo.append(j + i)
        self.state_dict = {st: i for st, i in
                           zip([''.join(x) for x in product(self.action_combo, repeat=1 * self.memory)],
                               list(range(6 ** self.memory)))}
        self.action_dict = {0: 'N', 1: 'W', 2: 'S'}

        # list of lists, each q_values[i] is one point on the q-values table
        self.q_values = [[0] for x in range((6 ** self.memory) * 3)]

    def play_turn(self, actions, op_actions):
        if len(actions) == 0:  # play random at first
            value = np.random.randint(3)
            if value == 0:
                return 'N'
            elif value == 1:
                return 'W'
            elif value == 2:
                return 'S'

        elif len(actions) < self.memory:  # play random at first
            value = np.random.randint(3)
            if value == 0:
                return 'N'
            elif value == 1:
                return 'W'
            elif value == 2:
                return 'S'

        else:
            state = []
            for i in reversed(range(self.memory)):  # state is made up of past i turns
                state.append(''.join([actions[-(i + 1)], op_actions[-(i + 1)]]))
            state = ''.join(state)

            if actions[-1] == 'N' and op_actions[-1] == 'C':
                reward = SignalGame.PAYOFF_1[0, 0]
            elif actions[-1] == 'W' and op_actions[-1] == 'C':
                reward = SignalGame.PAYOFF_1[1, 0]
            elif actions[-1] == 'S' and op_actions[-1] == 'C':
                reward = SignalGame.PAYOFF_1[2, 0]
            elif actions[-1] == 'N' and op_actions[-1] == 'D':
                reward = SignalGame.PAYOFF_1[0, 1]
            elif actions[-1] == 'W' and op_actions[-1] == 'D':
                reward = SignalGame.PAYOFF_1[1, 1]
            elif actions[-1] == 'S' and op_actions[-1] == 'D':
                reward = SignalGame.PAYOFF_1[2, 1]

            if len(actions) < EXPLORATION:  # explore (play random) for first x turns
                action = np.random.randint(3)
                return self.action_dict[action]


            else:
                # Play randomly in a certain proportion of moves - this proportion decays
                random_int = np.random.randint(len(actions) / EXP_N)
                if random_int == 1:
                    action = np.random.randint(3)
                else:
                    # Choose randomly if both q-values are equal
                    action = np.random.choice(
                        np.where(self.q_table[self.state_dict[state]] == self.q_table[self.state_dict[state]].max())[0])
                return self.action_dict[action]
            
class CyclerNWS(Player):
    """Cycles through moves NWS"""

    def __init__(self):
        pass

    def play_turn(self, actions, op_actions):
        if len(actions) == 0:
            return 'N'
        elif actions[-1].upper() == 'N':
            return 'W'
        elif actions[-1].upper() == 'W':
            return 'S'
        elif actions[-1].upper() == 'S':
            return 'N'
        
class CyclerNSW(Player):
    """Cycles through moves NWS"""

    def __init__(self):
        pass

    def play_turn(self, actions, op_actions):
        if len(actions) == 0:
            return 'N'
        elif actions[-1].upper() == 'N':
            return 'S'
        elif actions[-1].upper() == 'S':
            return 'W'
        elif actions[-1].upper() == 'W':
            return 'N'
        
        
            
class Random_Benef(Player):
    """Randomly selects to play N, W or S"""

    def __init__(self):
        pass

    def play_turn(self, actions, op_actions):
        value = np.random.randint(3)
        if value == 0:
            return 'N'
        elif value == 1:
            return 'W'
        elif value == 2:
            return 'S'
        
        
            
# PLAYER 2

class DonorQLearner(QLearner):
    """Learns the best strategy through Q-learning.
    Explores (plays randomly) for the first x (EXPLORATION variable) turns"""

    def __init__(self, memory=1):
        self.learning_rate = LEARNING_RATE
        self.discount_rate = DISCOUNT_RATE

        self.memory = memory
        self.q_table = np.zeros([6 ** self.memory, 2])  # q-table initialised with zeros

        # all possible action combinations for the states
        self.action_combo = []
        for i in ['C', 'D']:
            for j in ['N', 'W', 'S']:
                self.action_combo.append(i + j)
        self.state_dict = {st: i for st, i in
                           zip([''.join(x) for x in product(self.action_combo, repeat=1 * self.memory)],
                               list(range(6 ** self.memory)))}
        self.action_dict = {0: 'C', 1: 'D'}

        # list of lists, each q_values[i] is one point on the q-values table
        self.q_values = [[0] for x in range((6 ** self.memory) * 2)]

    def play_turn(self, actions, op_actions):
        if len(actions) == 0:  # play random at first
            value = np.random.randint(2)
            if value == 0:
                return 'C'
            elif value == 1:
                return 'D'

        elif len(actions) < self.memory:  # play random at first

            value = np.random.randint(2)
            if value == 0:
                return 'C'
            elif value == 1:
                return 'D'


        else:
            state = []
            for i in reversed(range(self.memory)):  # state is made up of past i turns
                state.append(''.join([actions[-(i + 1)], op_actions[-(i + 1)]]))
            state = ''.join(state)

            if actions[-1] == 'C' and op_actions[-1] == 'N':
                reward = SignalGame.PAYOFF_2[0, 0]
            elif actions[-1] == 'C' and op_actions[-1] == 'W':
                reward = SignalGame.PAYOFF_2[1, 0]
            elif actions[-1] == 'C' and op_actions[-1] == 'S':
                reward = SignalGame.PAYOFF_2[2, 0]
            elif actions[-1] == 'D' and op_actions[-1] == 'N':
                reward = SignalGame.PAYOFF_2[0, 1]
            elif actions[-1] == 'D' and op_actions[-1] == 'W':
                reward = SignalGame.PAYOFF_2[1, 1]
            elif actions[-1] == 'D' and op_actions[-1] == 'S':
                reward = SignalGame.PAYOFF_2[2, 1]

            if len(actions) < EXPLORATION:  # explore (play random) for first x turns
                action = np.random.randint(2)
                return self.action_dict[action]


            else:
                # Play randomly in a certain proportion of moves - this proportion decays
                random_int = np.random.randint(len(actions) / EXP_N)
                if random_int == 1:
                    action = np.random.randint(2)
                else:
                    # Choose randomly if both q-values are equal
                    action = np.random.choice(
                        np.where(self.q_table[self.state_dict[state]] == self.q_table[self.state_dict[state]].max())[0])

                return self.action_dict[action]