import seaborn as sns
import pandas as pd

def plotSeaborn(list1, list2):
    """
    :param list1: List of Original values
    :param list2: List of NN predicted values
    :return: None
    """

    df = pd.DataFrame()
    df['High level calculated energies (Ha)'] = list1
    df['NN predicted energies (Ha)'] = list2
    lm = sns.lmplot('High level calculated energies (Ha)','NN predicted energies (Ha)', data=df, scatter_kws={"s": 20, "alpha": 0.6}, line_kws={"alpha":0.5})
    lm.set(ylim=(-1.90, -1.78))
    lm.set(xlim=(-1.90, -1.78))

    sns.plt.show()