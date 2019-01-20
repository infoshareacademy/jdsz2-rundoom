import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

a = 123
m = 321

rand = [80, 81, 83, 80, 52, 69, 88]
linear = [90,99, 97, 91, 93, 91, 90]
index = ['10', '100', '1000', '10000', '100000', '100000', '1000000']
df = pd.DataFrame({'Random': rand,
                   'LCG': linear}, index=index)
plot_title = 'Random generator vs LCG (a={}, m={})'.format(a, m)
# df.plot()
ax = df.plot.bar(rot=0)
ax.set_title(plot_title)

ax.legend(loc='best')
plt.xlabel('number of generated numbers')
plt.ylabel('score in %')
plt.show()

