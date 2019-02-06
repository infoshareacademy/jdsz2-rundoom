import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import datetime

# data=np.random.uniform(size=1000)
# df=stats.kstest(data,'uniform')
# pval = df.pvalue
#
# print(pval)


def genmlcg(a, m,ile_liczb):
    dane = []
    x0 = datetime.datetime.now().microsecond- datetime.datetime.now().microsecond // 100 * 100
    poprzedni = x0
    for i in range(ile_liczb):
        xj = (a*poprzedni)%m
        poprzedni = xj
        xj = xj/m
        dane.append(xj)
    return dane


def generator_test(ile_liczb,ile_testow,a=43,m=801):
    generator_mlcg_licznik = 0
    generator_random_licznik = 0

    try:
        for i in range(1,ile_testow):
            generator_mlcg = genmlcg(a, m, ile_liczb=ile_liczb)
            generator_random = np.random.uniform(size=ile_liczb)
            test_mlcg = stats.kstest(generator_mlcg,'uniform')
            test_random = stats.kstest(generator_random,'uniform')
            #print(test_mlcg.pvalue, test_random.pvalue)
            generator_mlcg_licznik += test_mlcg.pvalue>0.05
            generator_random_licznik += test_random.pvalue>0.05
            print(generator_mlcg)
        print("H0 potwierdzone dla {} % wyników generatora mlcg a dla generatora liczb random wynosi {} %".format(generator_mlcg_licznik/ile_testow*100,generator_random_licznik/ile_testow*100))
    except:
        pass

generator_test(1000,100)


