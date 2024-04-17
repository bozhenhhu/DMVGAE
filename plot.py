import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
 

x = np.linspace(0,1,100000)
y_1 = stats.t.pdf(x,0.01)
y_2 = stats.t.pdf(x,0.1)
y_3 = stats.t.pdf(x,1)
# y_4 = stats.t.pdf(x,10)
# y_5 = stats.t.pdf(x,100)
plt.plot(x,y_1,label=r'$\nu=0.01$')
plt.plot(x,y_2,label=r'$\nu=0.1$')
plt.plot(x,y_3,label=r'$\nu=1$')
# plt.plot(x,y_4,label=r'$\nu=10$')
# plt.plot(x,y_5,label=r'$\nu=100$')
plt.title('t-distribution')
plt.legend(loc='upper right')

plt.tight_layout()
# plt.show()
plt.savefig("./t-distribution-1",dpi=300)

