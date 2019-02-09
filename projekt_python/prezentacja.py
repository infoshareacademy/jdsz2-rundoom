import numpy as np
import scipy.stats as stats
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def genmlcg(a, m, ile_liczb):
    """Generator MLCG"""
    dane = []
    x0 = datetime.datetime.now().microsecond - datetime.datetime.now().microsecond // 100 * 100
    poprzedni = x0
    for i in range(ile_liczb):
        xj = (a*poprzedni) % m
        poprzedni = xj
        xj = xj/m
        dane.append(xj)
    return dane

def wykresy():
    """Wykresy"""

    # ------------ Wykresy dla 100 liczb -----------------

    mlcg_100 = genmlcg(39373, 2147483647, 100)
    sns.set()
    plt.hist(mlcg_100, bins=10)
    plt.title('Generator MLCG - 100 liczb')
    plt.xlabel('liczby w przedziałach co 0.1')
    plt.ylabel('ilość liczb w przedziale')
    plt.show()

    random_100 = np.random.uniform(size=100)
    plt.hist(random_100, bins=10)
    plt.title('Generator z pakietu numpy - 100 liczb')
    plt.xlabel('liczby w przedziałach co 0.1')
    plt.ylabel('ilość liczb w przedziale')
    plt.show()

    # ----------- Porównanie Generator vs Numpy - Seaborn - 100 liczb

    porownanie_100 = pd.DataFrame({ 'Generator': np.concatenate([np.array(['MLCG'] * 100), np.array(['Numpy'] * 100)]),
                                    'Liczby': np.concatenate([np.array(mlcg_100), random_100])})
    plt.title('Generator vs Numpy - 100 liczb')
    sns.swarmplot(x='Generator', y='Liczby', data=porownanie_100)
    plt.show()

    # ------------ Wykresy dla 1000 liczb -----------------

    mlcg_1000 = genmlcg(39373, 2147483647, 1000)
    sns.set()
    plt.hist(mlcg_1000, bins=10)
    plt.title('Generator MLCG - 1000 liczb')
    plt.xlabel('liczby w przedziałach co 0.1')
    plt.ylabel('ilość liczb w przedziale')
    plt.show()

    random_1000 = np.random.uniform(size=1000)
    plt.hist(random_1000, bins=10)
    plt.title('Generator z pakietu numpy - 1000 liczb')
    plt.xlabel('liczby w przedziałach co 0.1')
    plt.ylabel('ilość liczb w przedziale')
    plt.show()

    # ------------ Empirical cumulative distribution function (ECDF) 1000 liczb

    y_1000 = np.arange(1, 1001) /1000
    mlcg_1000_ecdf = np.sort(mlcg_1000)
    random_1000_ecdf = np.sort(random_1000)
    plt.plot(mlcg_1000_ecdf, y_1000, marker='.', linestyle='none')
    plt.plot(random_1000_ecdf, y_1000, marker='.', linestyle='none')
    plt.title('ECDF 1000 liczb')
    plt.legend(('MLCG', 'Numpy'), loc='lower right')
    plt.xlabel('liczby')
    plt.ylabel('ECDF')
    plt.show()

    # ------------ Wykresy dla 10000 liczb -----------------

    mlcg_10000 = genmlcg(39373, 2147483647, 10000)
    sns.set()
    plt.hist(mlcg_10000, bins=10)
    plt.title('Generator MLCG - 10000 liczb')
    plt.xlabel('liczby w przedziałach co 0.1')
    plt.ylabel('ilość liczb w przedziale')
    plt.show()

    random_10000 = np.random.uniform(size=10000)
    plt.hist(random_10000, bins=10)
    plt.title('Generator z pakietu numpy - 10000 liczb')
    plt.xlabel('liczby w przedziałach co 0.1')
    plt.ylabel('ilość liczb w przedziale')
    plt.show()

    # ------------ Empirical cumulative distribution function (ECDF) 10000 liczb

    y_10000 = np.arange(1, 10001) /10000
    mlcg_10000_ecdf = np.sort(mlcg_10000)
    random_10000_ecdf = np.sort(random_10000)
    plt.plot(mlcg_10000_ecdf, y_10000, marker='.', linestyle='none')
    plt.plot(random_10000_ecdf, y_10000, marker='.', linestyle='none')
    plt.title('ECDF 10000 liczb')
    plt.legend(('MLCG', 'Numpy'), loc='lower right')
    plt.xlabel('liczby')
    plt.ylabel('ECDF')
    plt.show()

#wykresy()

def generator_test(ile_liczb, ile_testow, a=39373, m=2147483647, czy_print=True):
    """Test generatora MLCG i porównanie do generatora zaimplementowanego w Pythonie"""
    generator_mlcg_licznik = 0
    generator_random_licznik = 0

    try:
        for i in range(1, ile_testow):
            generator_mlcg = genmlcg(a, m, ile_liczb=ile_liczb)
            generator_random = np.random.uniform(size=ile_liczb)
            test_mlcg = stats.kstest(generator_mlcg, 'uniform')
            test_random = stats.kstest(generator_random, 'uniform')
            generator_mlcg_licznik += test_mlcg.pvalue > 0.05
            generator_random_licznik += test_random.pvalue > 0.05
        if czy_print:
            print("H0 potwierdzone dla {} % wyników generatora mlcg a dla generatora liczb random wynosi {} %".format(generator_mlcg_licznik/ile_testow*100,generator_random_licznik/ile_testow*100))
        return generator_mlcg_licznik / ile_testow * 100, generator_random_licznik/ile_testow*100
    except:
        pass

#generator_test(1000,1000)

def scenariusze():
    """Scenariusze dla różnych a i m"""
    a = np.array([[39373, 40014, 40692]] * 3).ravel(order='F')
    m = np.array([[2147483647, 2147483563, 2147483399]] * 3).ravel(order='F')
    ile_liczb = [100, 500, 1000] * int(len(a) / 3)
    ile_testow = [100, 500, 1000] * int(len(a) / 3)

    tabela = pd.DataFrame({'ile liczb' : ile_liczb,
                                'ile testów' : ile_testow,
                                'a' : a,
                                'm' : m,
                                'mlcg': None,
                                'random': None})

    for index, row in tabela.iterrows():
        tabela.iloc[index, 4] = generator_test(row[0], row[1], row[2], row[3], czy_print=False)[0]
        tabela.iloc[index, 5] = generator_test(row[0], row[1], row[2], row[3], czy_print=False)[1]

    return tabela

#print(scenariusze())

def scenariusze_wykresy():
    """Scenarusze z wykresami dla różnych a i m"""
    # --------- Scenariusz 1
    a1 = 39373
    m1 = 2147483647

    tabela1 = pd.DataFrame({'ile liczb' : np.array([100, 500, 1000]),
                            'ile testów' : np.array([100, 500, 1000]),
                            'a' : a1,
                            'm' : m1,
                            'mlcg': None,
                            'random': None})

    for index, row in tabela1.iterrows():
        tabela1.iloc[index, 4] = generator_test(row[0], row[1], row[2], row[3], czy_print=False)[0]
        tabela1.iloc[index, 5] = generator_test(row[0], row[1], row[2], row[3], czy_print=False)[1]

    plot_title = 'Random generator vs LCG (a={}, m={})'.format(a1, m1)
    ax = tabela1[['mlcg', 'random']].plot.bar(rot=0)
    ax.set_title(plot_title)

    ax.legend(loc='lower right')
    plt.xticks(ticks=[0, 1, 2], labels=['100', '500', '1000'])
    plt.xlabel('number of generated numbers')
    plt.ylabel('score in %')
    plt.show()

    print('\n',plot_title,'\n')
    print(tabela1)

    # --------- Scenariusz 2
    a1 = 40014
    m1 = 2147483563

    tabela1 = pd.DataFrame({'ile liczb' : np.array([100, 500, 1000]),
                            'ile testów' : np.array([100, 500, 1000]),
                            'a' : a1,
                            'm' : m1,
                            'mlcg': None,
                            'random': None})

    for index, row in tabela1.iterrows():
        tabela1.iloc[index, 4] = generator_test(row[0], row[1], row[2], row[3], czy_print=False)[0]
        tabela1.iloc[index, 5] = generator_test(row[0], row[1], row[2], row[3], czy_print=False)[1]

    plot_title = 'Random generator vs LCG (a={}, m={})'.format(a1, m1)
    ax = tabela1[['mlcg', 'random']].plot.bar(rot=0)
    ax.set_title(plot_title)

    ax.legend(loc='lower right')
    plt.xticks(ticks=[0, 1, 2], labels=['100', '500', '1000'])
    plt.xlabel('number of generated numbers')
    plt.ylabel('score in %')
    plt.show()

    print('\n',plot_title,'\n')
    print(tabela1)

    # --------- Scenariusz 3
    a1 = 40692
    m1 = 2147483399

    tabela1 = pd.DataFrame({'ile liczb': np.array([100, 500, 1000]),
                            'ile testów': np.array([100, 500, 1000]),
                            'a': a1,
                            'm': m1,
                            'mlcg': None,
                            'random': None})

    for index, row in tabela1.iterrows():
        tabela1.iloc[index, 4] = generator_test(row[0], row[1], row[2], row[3], czy_print=False)[0]
        tabela1.iloc[index, 5] = generator_test(row[0], row[1], row[2], row[3], czy_print=False)[1]

    plot_title = 'Random generator vs LCG (a={}, m={})'.format(a1, m1)
    ax = tabela1[['mlcg', 'random']].plot.bar(rot=0)
    ax.set_title(plot_title)

    ax.legend(loc='lower right')
    plt.xticks(ticks=[0, 1, 2], labels=['100', '500', '1000'])
    plt.xlabel('number of generated numbers')
    plt.ylabel('score in %')
    plt.show()

    print('\n', plot_title, '\n')
    print(tabela1)

#scenariusze_wykresy()




