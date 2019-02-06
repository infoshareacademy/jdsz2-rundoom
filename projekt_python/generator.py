import datetime
import matplotlib.pyplot as plt


#
# print('warunek a^2 < m ',a**2, m, (a**2) < m)
# q = m//a
# r = m % a
#
#
# print('q = ',q,'r = ',r)
#
# print('sprawdzenie czy m = a*q + r ', m, a*q + r)


# m = 2147483647
# a = 39373

# m = 8191
# a = 884

# m = 4294976291
# a = 1588635695

# m = 2147483563
# a = 40014


# m = 8001
# a = 43



dane = []

def generator(m,a):
    x0 = datetime.datetime.now().microsecond - datetime.datetime.now().microsecond // 100 * 100
    poprzedni = x0
    for i in range(2000):
        xj = (a*poprzedni)%m
        poprzedni = xj
        xj = xj/m
        dane.append(xj)


    print(dane)
    print(max(dane))

    plt.hist(dane,bins=100)

    plt.show()

generator(8001,43)

