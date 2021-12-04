import numpy as np
import random
from scipy.optimize import minimize, root
from scipy.misc import derivative
import matplotlib.pyplot as plt

def linear(param, b=0):
    global data
    f = param[0] * data[:, 0] + param[1]
    if b:
        return f
    return np.sum((f - data[:, 1]) ** 2)

def linear_lm(param, b=0):
    global data
    f = param[0] * data[:, 0] + param[1]
    if b:
        return f
    return (np.sum((f - data[:, 1]) ** 2), 0)

def rac(param, b=0):
    global data
    f = param[0] / (1 + param[1] * data[:, 0])
    if b:
        return f
    return np.sum((f - data[:, 1]) ** 2)

def rac_lm(param, b=0):
    global data
    f = param[0] / (1 + param[1] * data[:, 0])
    if b:
        return f
    return np.sum((f - data[:, 1]) ** 2), 0

def lineara(a):
    global data
    global A, B
    return np.sum(((a * data[:, 0] + B) - data[:, 1]) ** 2)

def linearb(b):
    global data
    global A, B
    return np.sum(((A * data[:, 0] + b) - data[:, 1]) ** 2)

def racA(a):
    global data
    global A, B
    return np.sum(((a / (1 + B * data[:, 0])) - data[:, 1]) ** 2)

def racB(b):
    global data
    global A, B
    return np.sum(((A / (1 + b * data[:, 0])) - data[:, 1]) ** 2)

def gradient(eps, fun, funA, funB):
    global A, B
    A, a0 = 0.01, 0.01
    B, b0 = 0.01, 0.01
    betaA, betaB = 0.01, 0.01
    d0 = fun([a0, b0])
    derA0 = derivative(funA, A)
    derB0 = derivative(funB, B)
    A = A - betaA * derA0
    B = B - betaB * derB0
    d = fun([A, B])

    while abs(d - d0) >= eps:

        derA = derA0
        derB = derB0
        derA0 = derivative(funA, A)
        derB0 = derivative(funB, B)
        betaA, betaB = abs((np.array([a0, b0]) - np.array([A, B])) * (np.array([derA, derB]) - np.array([derA0, derB0]))) / ((np.array([derA, derB]) * np.array([derA0, derB0])))
        a0 = A
        b0 = B
        A = A - betaA * derA0
        B = B - betaB * derB0
        d0 = d
        d = fun([A, B])
    return (A, B)

## Part II linear
alpha, beta = random.random(), random.random()
eps = 0.001

deltaK = np.array([random.gauss(0,1) for i in range(0,100)])
xk = np.arange(100)/100
yk = alpha * xk + beta + deltaK
data = np.transpose(np.array([xk,yk]))

x0=np.array([0,0])
grad = gradient(eps,linear,lineara,linearb)
cg = minimize(linear, x0, method='CG', options={'xtol': 1e-3, 'disp': True})
ncg = minimize(linear, x0, method='TNC', options={'xtol': 1e-3, 'disp': True})
lm = root(linear_lm, x0, method='lm')
fig1 = plt.figure()
ax = plt.axes()
ax.plot(xk,yk)
ax.plot(xk,linear(cg.x,1),label="Сопряженный градиент")
ax.plot(xk,linear(grad,1),label="Градиент")
ax.plot(xk,linear(ncg.x,1),label="Ньютон")
ax.plot(xk,linear(lm.x,1),label=" Левенберг - Марквардт")
plt.show()
ax.set_title('Линейная аппроксимация')
ax.legend()
plt.savefig('lin1.png', dpi = 1000)
## Part II Дробно рац

x0=np.array([0,0])
grad_rac = gradient(eps,rac,racA,racB)
cg_rac = minimize(rac, x0, method='CG', options={'xtol': 1e-3, 'disp': True})
ncg_rac = minimize(rac, x0, method='TNC', options={'xtol': 1e-3, 'disp': True})
lm_rac = root(rac_lm, x0, method='lm')
fig2 = plt.figure()
ax = plt.axes()
ax.plot(xk,yk)
ax.plot(xk,rac(cg_rac.x,1),label="Сопряженный градиент")
ax.plot(xk,rac(grad_rac,1),label="Градиент")
ax.plot(xk,rac(ncg_rac.x,1),label="Ньютон")
ax.plot(xk,rac(lm_rac.x,1),label=" Левенберг - Марквардт")
plt.show()
ax.set_title('Рациональная аппроксимация')
ax.legend()
plt.savefig('ratio1.png', dpi = 1000)



