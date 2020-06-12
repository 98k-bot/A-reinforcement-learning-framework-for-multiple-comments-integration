import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

group_prefix = "group1"

if group_prefix == "group1":
                 #MMS      #SOSML
    ROUGE1 =    [0.401302, 0.408784, 0.39, 0.39, 0.39, 0.39]
    ROUGE2 =    [0.196034, 0.218321, 0.19, 0.19, 0.19, 0.19]
    ROUGEL =    [0.372016, 0.384031, 0.3, 0.3, 0.3, 0.3]
    SENTI_AVG = [5.290285, 4.693471, 4.0, 4.0, 4.0, 4.0]
    SENTI_MAX = [9.762570, 9.496670, 9.3, 9.3, 9.3, 9.3]
    SSECIF = {"R1":0.409494,  "R2":0.211660,  "RL":0.383720, "SENTI_AVG":5.329823, "HUMAN_AVG":3.776976, "SENTI_MAX":9.809613, "HUMAN_MAX":8.240730}

if group_prefix == "group2":
    pass

method_names = ['MMS_Text', 'SOSML', 'ASRL', 'SummaRuNNer', 'DQN_RNN', 'Reaper']


def make_ROUGE(axs, names, values, plot_val, y_label=None, title=None):
    # SSECIF
    axs.plot(names, [plot_val]*len(names), linestyle="dotted", marker="v", ms=7, lw=2, alpha=0.9, label="SSECIF_Full") #, alpha=0.7, mfc='orange'

    # others
    axs.plot(names, values, linestyle="dashed", marker="^", ms=7, lw=2, alpha=0.9, label="Others")
    axs.set_ylabel(y_label)
    axs.set_title(title)
    axs.grid(linestyle='--')

    for tick in axs.get_xticklabels():
        tick.set_rotation(20)

def make_Senti(axs, names, values, agent_val, human_val, y_label=None, title=None):

    # agent
    axs.plot(names, [agent_val]*len(names), linestyle="solid", marker="v", ms=7, lw=2, alpha=0.9, label="SSECIF_Full") #, alpha=0.7, mfc='orange'
    # human
    axs.plot(names, [human_val]*len(names), linestyle="dotted", color='red',marker="P", ms=7, lw=2, alpha=0.9, label="Human") #, alpha=0.7, mfc='orange'

    # others
    axs.plot(names, values, linestyle="dashed", marker="^", ms=7, lw=2, alpha=0.9, label="Others")
    axs.set_ylabel(y_label)
    axs.set_title(title)
    axs.grid(linestyle='--')

    for tick in axs.get_xticklabels():
        tick.set_rotation(20)


if __name__ == "__main__":

    show_R, show_Senti = 0, 1

    if show_R:
        others_data = ROUGEL
        data_key = "RL"
        y_label = "ROUGE_L"
        title = "ROUGE_L Comparison"

        fig, axs = plt.subplots(1, 1)
        make_ROUGE(axs, method_names, others_data, SSECIF[data_key], y_label=y_label, title=title)

    if show_Senti:
        others_data = SENTI_MAX
        agent_key = "SENTI_MAX"
        human_key = "HUMAN_MAX"
        y_label = "Max_Sentiment"
        title = "Max Sentiment Comparison"

        fig, axs = plt.subplots(1, 1)
        make_Senti(axs, method_names, others_data, SSECIF[agent_key], SSECIF[human_key], y_label=y_label, title=title)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(5, 5, forward=True)
    plt.legend()
    plt.show()




