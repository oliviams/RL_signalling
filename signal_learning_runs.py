import numpy as np
import pandas as pd
from itertools import product
from matplotlib.backends.backend_pdf import PdfPages

from RL_signalling.signal_learning import *


# PLOTTING MULTIPLE RUN RESULTS

# Variables to change
memory = 1
runs = 1000
rounds = 500
signals = ['N', 'W', 'S']
colours = ['blue', 'orange', 'green']
actions_list = ['C', 'D']
colours_act = ['blue', 'red']

action_combo = [j + i for i in actions_list for j in signals]

state_names = [''.join(x) for x in product(action_combo, repeat=1 * memory)]

# Creating lists of data to plot
q_values = []
op_q_values = []
actions = []
op_actions = []
cumulative = []
strategies = []

for i in range(runs):
    game = PSGame(SignalQLearner(memory=memory), DonorQLearner(memory=memory), rounds = rounds)
    if 'QLearner' in type(game.player).__name__:
        if 'QLearner' in type(game.opponent).__name__:
            q_values.append(game.q_values_list()[0])
            actions.append(game.ACTIONS[-100:])
            cumulative.append(game.cumulative_result())
            strategies.append(game.resulting_strategies())
            
            op_q_values.append(game.q_values_list()[1])
            op_actions.append(game.OP_ACTIONS[-100:])
            
        else:
            q_values.append(game.q_values_list())
            actions.append(game.ACTIONS[-100:])
            cumulative.append(game.cumulative_result())
            strategies.append(game.resulting_strategies())
            
    elif 'QLearner' in type(game.opponent).__name__:
        op_q_values.append(game.q_values_list())
        op_actions.append(game.OP_ACTIONS[-100:])
        cumulative.append(game.cumulative_result())
        strategies.append(game.resulting_strategies())

        
# Accounting for whether one or both players are q-learners

if 'QLearner' in type(game.player).__name__:
    if 'QLearner' in type(game.opponent).__name__:
        # Q-value averages
        states = []    
        for j in range(len(signals)*memory*6):
            lists = []
            for i in range(runs):
                lists.append(q_values[i][j])
            states.append(lists)
    
        pdf = PdfPages('Replication/MS/S_' + str(S) + '_V_' + str(V) + '_r_' + str(r) + '_' + thirst + '/' + type(game.player).__name__ + '_' + type(game.opponent).__name__ + '_LR' +
               str(LEARNING_RATE) + '_DR' + str(DISCOUNT_RATE) + '_M' + str(memory) + '_q-values_PLAYER1.pdf')
        
        for i in range(0, len(states), len(signals)):
            x =  np.arange(0, rounds+1, 1)
            plot = plt.figure()
            for k in range(len(signals)):
                array = np.array(states[i+k])
                mean = array.mean(axis=0)
                std = np.std(array, axis=0)
                plt.plot(mean, color=colours[k], lw=1, label = 'Action: ' + signals[k])
                plt.fill_between(x, (mean-std), (mean+std), color=colours[k], alpha=0.3)
                plt.legend()
                plt.ylim(0)
                pd.DataFrame(array).to_csv('Replication/MS/S_' + str(S) + '_V_' + str(V) + '_r_' + str(r) + '_' + thirst + '/' + type(game.player).__name__ + '_' + type(game.opponent).__name__ + '_LR' +
                                           str(LEARNING_RATE) + '_DR' + str(DISCOUNT_RATE) + '_M' + str(memory) + 
                                           '_q-values_P1_' + signals[k] + '.csv', mode='a', header=False)
            plt.ylabel('State: ' + state_names[int(i/3)])
            pdf.savefig(plot)

        pdf.close()

        # Cumulative result averages
        player_1 = np.array([[x[0] for x in i] for i in cumulative])
        player_2 = np.array([[x[1] for x in i] for i in cumulative])
        pd.DataFrame(player_1).to_csv('Replication/MS/S_' + str(S) + '_V_' + str(V) + '_r_' + str(r) + '_' + thirst + '/' + type(game.player).__name__ + '_' + type(game.opponent).__name__ + '_LR' +
       str(LEARNING_RATE) + '_DR' + str(DISCOUNT_RATE) + '_M' + str(memory) + '_cumulative-score_p1.csv') 
        pd.DataFrame(player_1).to_csv('Replication/MS/S_' + str(S) + '_V_' + str(V) + '_r_' + str(r) + '_' + thirst + '/' + type(game.player).__name__ + '_' + type(game.opponent).__name__ + '_LR' +
       str(LEARNING_RATE) + '_DR' + str(DISCOUNT_RATE) + '_M' + str(memory) + '_cumulative-score_p2.csv') 

        mean1 = player_1.mean(axis=0)
        std1 = np.std(player_1, axis=0)
        mean2 = player_2.mean(axis=0)
        std2 = np.std(player_2, axis=0)
        x =  np.arange(0, rounds, 1)

        plt.figure()
        plt.xlabel("Round")
        plt.ylabel("Score")
        plt.title("Cumulative scores")

        plt.plot(x, mean1, label=type(game.player).__name__, color = 'blue')
        plt.plot(x, mean2, label=type(game.opponent).__name__, color = 'red')
        plt.fill_between(x, (mean1-std1), (mean1+std1), color='blue', alpha=0.3)
        plt.fill_between(x, (mean2-std2), (mean2+std2), color='red', alpha=0.3)
        plt.legend()
        plt.savefig('Replication/MS/S_' + str(S) + '_V_' + str(V) + '_r_' + str(r) + '_' + thirst + '/' + type(game.player).__name__ + '_' + type(game.opponent).__name__ + '_LR' +
               str(LEARNING_RATE) + '_DR' + str(DISCOUNT_RATE) + '_M' + str(memory) + '_cumulative-score.pdf')
        
        # Resulting strategies (if there is a 95% match in the rows, they are grouped)
        df = pd.DataFrame(actions, columns = range(rounds-99, rounds+1), index=range(1, runs+1))
        df_2 = pd.DataFrame(strategies, columns = [type(game.player).__name__, type(game.opponent).__name__], index=range(1, runs+1))
        
        strat = pd.DataFrame(df_2[type(game.player).__name__].value_counts())
        strat.index.name = type(game.player).__name__
        strat.columns = ['Frequency']
        strat = strat.reset_index()
        strat.to_csv('Replication/MS/S_' + str(S) + '_V_' + str(V) + '_r_' + str(r) + '_' + thirst + '/' + type(game.player).__name__ + '_' + type(game.opponent).__name__ + '_LR' +
               str(LEARNING_RATE) + '_DR' + str(DISCOUNT_RATE) + '_M' + str(memory) + '_strategy-summary_P1.csv')

        plt.figure()
        plt.barh(strat[type(game.player).__name__], strat['Frequency'])
        plt.xlabel("Frequency")
        plt.ylabel("Strategy")
        plt.title("Frequency of occurrence of resulting strategies")
        plt.savefig('Replication/MS/S_' + str(S) + '_V_' + str(V) + '_r_' + str(r) + '_' + thirst + '/' + type(game.player).__name__ + '_' + type(game.opponent).__name__ + '_LR' +
               str(LEARNING_RATE) + '_DR' + str(DISCOUNT_RATE) + '_M' + str(memory) + '_resulting-strategies_P1.pdf')

        freq = []
        index_list = []

        df_copy = df.copy()

        for i in range(len(df_copy)):
            try:
                freq.append(len(df_copy[df_copy.eq(df_copy[:1].values.tolist()[0]).sum(axis=1).ge(95)]))
                index_list.append(df_copy[df_copy.eq(df_copy[:1].values.tolist()[0]).sum(axis=1).ge(95)].index.to_list())
                df_copy = df_copy.drop(df_copy[df_copy.eq(df_copy[:1].values.tolist()[0]).sum(axis=1).ge(95)].index.to_list())
            except:
                pass

        new_df = df.loc[[i[0] for i in index_list]]

        new_df['Frequency'] = freq
        new_df = new_df.style.applymap(signal_colours)
        new_df.to_excel('Replication/MS/S_' + str(S) + '_V_' + str(V) + '_r_' + str(r) + '_' + thirst + '/' + type(game.player).__name__ + '_' + type(game.opponent).__name__ + '_LR' +
               str(LEARNING_RATE) + '_DR' + str(DISCOUNT_RATE) + '_M' + str(memory) + '_strategies_PLAYER1.xlsx')
        
        states = []    
        for j in range(len(actions_list)*memory*6):
            lists = []
            for i in range(runs):
                lists.append(op_q_values[i][j])
            states.append(lists)

        pdf = PdfPages('Replication/MS/S_' + str(S) + '_V_' + str(V) + '_r_' + str(r) + '_' + thirst + '/' + type(game.player).__name__ + '_' + type(game.opponent).__name__ + '_LR' +
               str(LEARNING_RATE) + '_DR' + str(DISCOUNT_RATE) + '_M' + str(memory) + '_q-values_PLAYER2.pdf')

        for i in range(0, len(states), len(actions_list)):
            x =  np.arange(0, rounds+1, 1)
            plot = plt.figure()
            for k in range(len(actions_list)):
                array = np.array(states[i+k])
                mean = array.mean(axis=0)
                std = np.std(array, axis=0)
                plt.plot(mean, color=colours_act[k], lw=1, label = 'Action: ' + actions_list[k])
                plt.fill_between(x, (mean-std), (mean+std), color=colours_act[k], alpha=0.3)
                plt.legend()
                plt.ylim(0)
                pd.DataFrame(array).to_csv('Replication/MS/S_' + str(S) + '_V_' + str(V) + '_r_' + str(r) + '_' + thirst + '/' + type(game.player).__name__ + '_' + type(game.opponent).__name__ + '_LR' +
                                           str(LEARNING_RATE) + '_DR' + str(DISCOUNT_RATE) + '_M' + str(memory) + 
                                           '_q-values_P2_' + actions_list[k] + '.csv', mode='a', header=False)
            plt.ylabel('State: ' + state_names[int(i/2)])
            pdf.savefig(plot)

        pdf.close()

        df = pd.DataFrame(op_actions, columns = range(rounds-99, rounds+1), index=range(1, runs+1))
        df_2 = pd.DataFrame(strategies, columns = [type(game.player).__name__, type(game.opponent).__name__], index=range(1, runs+1))
        
        strat = pd.DataFrame(df_2[type(game.opponent).__name__].value_counts())
        strat.index.name = type(game.opponent).__name__
        strat.columns = ['Frequency']
        strat = strat.reset_index()
        strat.to_csv('Replication/MS/S_' + str(S) + '_V_' + str(V) + '_r_' + str(r) + '_' + thirst + '/' + type(game.player).__name__ + '_' + type(game.opponent).__name__ + '_LR' +
               str(LEARNING_RATE) + '_DR' + str(DISCOUNT_RATE) + '_M' + str(memory) + '_strategy-summary_P2.csv')

        plt.figure()
        plt.barh(strat[type(game.opponent).__name__], strat['Frequency'])
        plt.xlabel("Frequency")
        plt.ylabel("Strategy")
        plt.title("Frequency of occurrence of resulting strategies")
        plt.savefig('Replication/MS/S_' + str(S) + '_V_' + str(V) + '_r_' + str(r) + '_' + thirst + '/' + type(game.player).__name__ + '_' + type(game.opponent).__name__ + '_LR' +
               str(LEARNING_RATE) + '_DR' + str(DISCOUNT_RATE) + '_M' + str(memory) + '_resulting-strategies_P2.pdf')

        freq = []
        index_list = []

        df_copy = df.copy()

        for i in range(len(df_copy)):
            try:
                freq.append(len(df_copy[df_copy.eq(df_copy[:1].values.tolist()[0]).sum(axis=1).ge(95)]))
                index_list.append(df_copy[df_copy.eq(df_copy[:1].values.tolist()[0]).sum(axis=1).ge(95)].index.to_list())
                df_copy = df_copy.drop(df_copy[df_copy.eq(df_copy[:1].values.tolist()[0]).sum(axis=1).ge(95)].index.to_list())
            except:
                pass

        new_df = df.loc[[i[0] for i in index_list]]

        new_df['Frequency'] = freq
        new_df = new_df.style.applymap(signal_colours)
        new_df.to_excel('Replication/MS/S_' + str(S) + '_V_' + str(V) + '_r_' + str(r) + '_' + thirst + '/' + type(game.player).__name__ + '_' + type(game.opponent).__name__ + '_LR' +
               str(LEARNING_RATE) + '_DR' + str(DISCOUNT_RATE) + '_M' + str(memory) + '_strategies_PLAYER2.xlsx')
        
    else:
        states = []    
        for j in range(len(signals)*memory*6):
            lists = []
            for i in range(runs):
                lists.append(q_values[i][j])
            states.append(lists)
    
        pdf = PdfPages('Replication/MS/S_' + str(S) + '_V_' + str(V) + '_r_' + str(r) + '_' + thirst + '/' + type(game.player).__name__ + '_' + type(game.opponent).__name__ + '_LR' +
               str(LEARNING_RATE) + '_DR' + str(DISCOUNT_RATE) + '_M' + str(memory) + '_q-values.pdf')

        for i in range(0, len(states), len(signals)):
            x =  np.arange(0, rounds+1, 1)
            plot = plt.figure()
            for k in range(len(signals)):
                array = np.array(states[i+k])
                mean = array.mean(axis=0)
                std = np.std(array, axis=0)
                plt.plot(mean, color=colours[k], lw=1, label = 'Action: ' + signals[k])
                plt.fill_between(x, (mean-std), (mean+std), color=colours[k], alpha=0.3)
                plt.legend()
                plt.ylim(0)
                pd.DataFrame(array).to_csv('Replication/MS/S_' + str(S) + '_V_' + str(V) + '_r_' + str(r) + '_' + thirst + '/' + type(game.player).__name__ + '_' + type(game.opponent).__name__ + '_LR' +
                                           str(LEARNING_RATE) + '_DR' + str(DISCOUNT_RATE) + '_M' + str(memory) + 
                                           '_q-values_P1_' + signals[k] + '.csv', mode='a', header=False)
            plt.ylabel('State: ' + state_names[int(i/3)])
            pdf.savefig(plot)



        pdf.close()


        player_1 = np.array([[x[0] for x in i] for i in cumulative])
        player_2 = np.array([[x[1] for x in i] for i in cumulative])
        pd.DataFrame(player_1).to_csv('Replication/MS/S_' + str(S) + '_V_' + str(V) + '_r_' + str(r) + '_' + thirst + '/' + type(game.player).__name__ + '_' + type(game.opponent).__name__ + '_LR' +
       str(LEARNING_RATE) + '_DR' + str(DISCOUNT_RATE) + '_M' + str(memory) + '_cumulative-score_p1.csv') 
        pd.DataFrame(player_1).to_csv('Replication/MS/S_' + str(S) + '_V_' + str(V) + '_r_' + str(r) + '_' + thirst + '/' + type(game.player).__name__ + '_' + type(game.opponent).__name__ + '_LR' +
       str(LEARNING_RATE) + '_DR' + str(DISCOUNT_RATE) + '_M' + str(memory) + '_cumulative-score_p2.csv')

        mean1 = player_1.mean(axis=0)
        std1 = np.std(player_1, axis=0)
        mean2 = player_2.mean(axis=0)
        std2 = np.std(player_2, axis=0)
        x =  np.arange(0, rounds, 1)

        plt.figure()
        plt.xlabel("Round")
        plt.ylabel("Score")
        plt.title("Cumulative scores")

        plt.plot(x, mean1, label=type(game.player).__name__, color = 'blue')
        plt.plot(x, mean2, label=type(game.opponent).__name__, color = 'red')
        plt.fill_between(x, (mean1-std1), (mean1+std1), color='blue', alpha=0.3)
        plt.fill_between(x, (mean2-std2), (mean2+std2), color='red', alpha=0.3)
        plt.legend()
        plt.savefig('Replication/MS/S_' + str(S) + '_V_' + str(V) + '_r_' + str(r) + '_' + thirst + '/' + type(game.player).__name__ + '_' + type(game.opponent).__name__ + '_LR' +
               str(LEARNING_RATE) + '_DR' + str(DISCOUNT_RATE) + '_M' + str(memory) + '_cumulative-score.pdf')
        
        df = pd.DataFrame(actions, columns = range(rounds-99, rounds+1), index=range(1, runs+1))
        df_2 = pd.DataFrame(strategies, columns = [type(game.player).__name__, type(game.opponent).__name__], index=range(1, runs+1))
        
        strat = pd.DataFrame(df_2[type(game.player).__name__].value_counts())
        strat.index.name = type(game.player).__name__
        strat.columns = ['Frequency']
        strat = strat.reset_index()
        strat.to_csv('Replication/MS/S_' + str(S) + '_V_' + str(V) + '_r_' + str(r) + '_' + thirst + '/' + type(game.player).__name__ + '_' + type(game.opponent).__name__ + '_LR' +
               str(LEARNING_RATE) + '_DR' + str(DISCOUNT_RATE) + '_M' + str(memory) + '_strategy-summary_P1.csv')

        plt.figure()
        plt.barh(strat[type(game.player).__name__], strat['Frequency'])
        plt.xlabel("Frequency")
        plt.ylabel("Strategy")
        plt.title("Frequency of occurrence of resulting strategies")
        plt.savefig('Replication/MS/S_' + str(S) + '_V_' + str(V) + '_r_' + str(r) + '_' + thirst + '/' + type(game.player).__name__ + '_' + type(game.opponent).__name__ + '_LR' +
               str(LEARNING_RATE) + '_DR' + str(DISCOUNT_RATE) + '_M' + str(memory) + '_resulting-strategies_P1.pdf')

        freq = []
        index_list = []

        df_copy = df.copy()

        for i in range(len(df_copy)):
            try:
                freq.append(len(df_copy[df_copy.eq(df_copy[:1].values.tolist()[0]).sum(axis=1).ge(95)]))
                index_list.append(df_copy[df_copy.eq(df_copy[:1].values.tolist()[0]).sum(axis=1).ge(95)].index.to_list())
                df_copy = df_copy.drop(df_copy[df_copy.eq(df_copy[:1].values.tolist()[0]).sum(axis=1).ge(95)].index.to_list())
            except:
                pass

        new_df = df.loc[[i[0] for i in index_list]]

        new_df['Frequency'] = freq
        new_df = new_df.style.applymap(signal_colours)
        new_df.to_excel('Replication/MS/S_' + str(S) + '_V_' + str(V) + '_r_' + str(r) + '_' + thirst + '/' + type(game.player).__name__ + '_' + type(game.opponent).__name__ + '_LR' +
               str(LEARNING_RATE) + '_DR' + str(DISCOUNT_RATE) + '_M' + str(memory) + '_strategies.xlsx')
        
elif 'QLearner' in type(game.opponent).__name__:       
    states = []    
    for j in range(len(actions_list)*memory*6):
        lists = []
        for i in range(runs):
            lists.append(op_q_values[i][j])
        states.append(lists)

    pdf = PdfPages('Replication/MS/S_' + str(S) + '_V_' + str(V) + '_r_' + str(r) + '_' + thirst + '/' + type(game.player).__name__ + '_' + type(game.opponent).__name__ + '_LR' +
               str(LEARNING_RATE) + '_DR' + str(DISCOUNT_RATE) + '_M' + str(memory) + '_q-values.pdf')

    for i in range(0, len(states), len(actions_list)):
        x =  np.arange(0, rounds+1, 1)
        plot = plt.figure()
        for k in range(len(actions_list)):
            array = np.array(states[i+k])
            mean = array.mean(axis=0)
            std = np.std(array, axis=0)
            plt.plot(mean, color=colours_act[k], lw=1, label = 'Action: ' + actions_list[k])
            plt.fill_between(x, (mean-std), (mean+std), color=colours_act[k], alpha=0.3)
            plt.legend()
            plt.ylim(0)
            pd.DataFrame(array).to_csv('Replication/MS/S_' + str(S) + '_V_' + str(V) + '_r_' + str(r) + '_' + thirst + '/' + type(game.player).__name__ + '_' + type(game.opponent).__name__ + '_LR' +
                                           str(LEARNING_RATE) + '_DR' + str(DISCOUNT_RATE) + '_M' + str(memory) + 
                                           '_q-values_P2_' + actions_list[k] + '.csv', mode='a', header=False)
        plt.ylabel('State: ' + state_names[int(i/2)])
        pdf.savefig(plot)

    pdf.close()


    player_1 = np.array([[x[0] for x in i] for i in cumulative])
    player_2 = np.array([[x[1] for x in i] for i in cumulative])
    pd.DataFrame(player_1).to_csv('Replication/MS/S_' + str(S) + '_V_' + str(V) + '_r_' + str(r) + '_' + thirst + '/' + type(game.player).__name__ + '_' + type(game.opponent).__name__ + '_LR' +
       str(LEARNING_RATE) + '_DR' + str(DISCOUNT_RATE) + '_M' + str(memory) + '_cumulative-score_p1.csv') 
    pd.DataFrame(player_1).to_csv('Replication/MS/S_' + str(S) + '_V_' + str(V) + '_r_' + str(r) + '_' + thirst + '/' + type(game.player).__name__ + '_' + type(game.opponent).__name__ + '_LR' +
       str(LEARNING_RATE) + '_DR' + str(DISCOUNT_RATE) + '_M' + str(memory) + '_cumulative-score_p2.csv')

    mean1 = player_1.mean(axis=0)
    std1 = np.std(player_1, axis=0)
    mean2 = player_2.mean(axis=0)
    std2 = np.std(player_2, axis=0)
    x =  np.arange(0, rounds, 1)

    plt.figure()
    plt.xlabel("Round")
    plt.ylabel("Score")
    plt.title("Cumulative scores")

    plt.plot(x, mean1, label=type(game.player).__name__, color = 'blue')
    plt.plot(x, mean2, label=type(game.opponent).__name__, color = 'red')
    plt.fill_between(x, (mean1-std1), (mean1+std1), color='blue', alpha=0.3)
    plt.fill_between(x, (mean2-std2), (mean2+std2), color='red', alpha=0.3)
    plt.legend()
    plt.savefig('Replication/MS/S_' + str(S) + '_V_' + str(V) + '_r_' + str(r) + '_' + thirst + '/' + type(game.player).__name__ + '_' + type(game.opponent).__name__ + '_LR' +
               str(LEARNING_RATE) + '_DR' + str(DISCOUNT_RATE) + '_M' + str(memory) + '_cumulative-score.pdf')
    
    df = pd.DataFrame(op_actions, columns = range(rounds-99, rounds+1), index=range(1, runs+1))
    df_2 = pd.DataFrame(strategies, columns = [type(game.player).__name__, type(game.opponent).__name__], index=range(1, runs+1))
        
    strat = pd.DataFrame(df_2[type(game.opponent).__name__].value_counts())
    strat.index.name = type(game.opponent).__name__
    strat.columns = ['Frequency']
    strat = strat.reset_index()
    strat.to_csv('Replication/MS/S_' + str(S) + '_V_' + str(V) + '_r_' + str(r) + '_' + thirst + '/' + type(game.player).__name__ + '_' + type(game.opponent).__name__ + '_LR' +
           str(LEARNING_RATE) + '_DR' + str(DISCOUNT_RATE) + '_M' + str(memory) + '_strategy-summary_P2.csv')

    plt.figure()
    plt.barh(strat[type(game.opponent).__name__], strat['Frequency'])
    plt.xlabel("Frequency")
    plt.ylabel("Strategy")
    plt.title("Frequency of occurrence of resulting strategies")
    plt.savefig('Replication/MS/S_' + str(S) + '_V_' + str(V) + '_r_' + str(r) + '_' + thirst + '/' + type(game.player).__name__ + '_' + type(game.opponent).__name__ + '_LR' +
           str(LEARNING_RATE) + '_DR' + str(DISCOUNT_RATE) + '_M' + str(memory) + '_resulting-strategies_P2.pdf')


    freq = []
    index_list = []

    df_copy = df.copy()

    for i in range(len(df_copy)):
        try:
            freq.append(len(df_copy[df_copy.eq(df_copy[:1].values.tolist()[0]).sum(axis=1).ge(95)]))
            index_list.append(df_copy[df_copy.eq(df_copy[:1].values.tolist()[0]).sum(axis=1).ge(95)].index.to_list())
            df_copy = df_copy.drop(df_copy[df_copy.eq(df_copy[:1].values.tolist()[0]).sum(axis=1).ge(95)].index.to_list())
        except:
            pass

    new_df = df.loc[[i[0] for i in index_list]]

    new_df['Frequency'] = freq
    new_df = new_df.style.applymap(signal_colours)
    new_df.to_excel('Replication/MS/S_' + str(S) + '_V_' + str(V) + '_r_' + str(r) + '_' + thirst + '/' + type(game.player).__name__ + '_' + type(game.opponent).__name__ + '_LR' +
               str(LEARNING_RATE) + '_DR' + str(DISCOUNT_RATE) + '_M' + str(memory) + '_strategies.xlsx')
