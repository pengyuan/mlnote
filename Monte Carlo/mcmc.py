# coding: utf-8
from pylab import *

"""采样问题：分布->数据
计算机可以获得服从均匀分布的随机数（容易根据种子生成伪随机数）。
一般的采样问题，都可以理解成：有了均匀分布（简单分布）的采样，如何去获取复杂分布的采样。
"""

"""M-H采样
"""


def mh():
    mu = 3
    sigma = 10

    def qsample():
        return np.random.normal(mu, sigma)

    def q(x):
        return exp(-(x - mu) ** 2 / (sigma ** 2))

    def p(x):
        """目标分布"""
        return 0.3 * np.exp(-(x - 0.3) ** 2) + 0.7 * np.exp(-(x - 2.) ** 2 / 0.3)

    def hm(n=10000):
        sample = np.zeros(n)
        sample[0] = 0.5
        for i in range(n - 1):
            q_s = qsample()
            u = np.random.rand()
            if u < min(1, (p(q_s) * q(sample[i])) / (p(sample[i]) * q(q_s))):
                sample[i + 1] = q_s
            else:
                sample[i + 1] = sample[i]
        return sample

    x = np.arange(0, 4, 0.1)
    realdata = p(x)
    N = 10000
    sample = hm(N)
    plt.plot(x, realdata, 'g', lw=3)
    plt.plot(x, q(x), 'r')
    plt.hist(sample, bins=x, normed=1, fc='c')
    plt.show()


def m():
    # The Metropolis-Hastings algorithm
    def p(x):
        mu1 = 3
        mu2 = 10
        v1 = 10
        v2 = 3
        return 0.3 * exp(-(x - mu1) ** 2 / v1) + 0.7 * exp(-(x - mu2) ** 2 / v2)

    def q(x):
        mu = 5
        sigma = 10
        return exp(-(x - mu) ** 2 / (sigma ** 2))

    stepsize = 0.5
    x = arange(-10, 20, stepsize)
    px = zeros(shape(x))
    for i in range(len(x)):
        px[i] = p(x[i])
    N = 5000

    # independence chain
    u = np.random.rand(N)
    mu = 5
    sigma = 10
    y = zeros(N)
    y[0] = np.random.normal(mu, sigma)
    for i in range(N - 1):
        ynew = np.random.normal(mu, sigma)
        alpha = min(1, p(ynew) * q(y[i]) / (p(y[i]) * q(ynew)))
        if u[i] < alpha:
            y[i + 1] = ynew
        else:
            y[i + 1] = y[i]

    # random walk chain
    u2 = np.random.rand(N)
    sigma = 10
    y2 = zeros(N)
    y2[0] = np.random.normal(0, sigma)
    for i in range(N - 1):
        y2new = y2[i] + np.random.normal(0, sigma)
        alpha = min(1, p(y2new) / p(y2[i]))
        if u2[i] < alpha:
            y2[i + 1] = y2new
        else:
            y2[i + 1] = y2[i]

    figure(1)
    nbins = 30
    hist(y, bins=x)
    plot(x, px * N / sum(px), color='g', linewidth=2)
    plot(x, q(x) * N / sum(px), color='r', linewidth=2)

    figure(2)
    nbins = 30
    hist(y2, bins=x)
    plot(x, px * N / sum(px), color='g', linewidth=2)

    show()


"""Gibbs采样
"""


def g():
    def pXgivenY(y, m1, m2, s1, s2):
        return np.random.normal(m1 + (y - m2) / s2, s1)

    def pYgivenX(x, m1, m2, s1, s2):
        return np.random.normal(m2 + (x - m1) / s1, s2)

    def gibbs(N=5000):
        k = 20
        x0 = np.zeros(N, dtype=float)
        m1 = 10
        m2 = 20
        s1 = 2
        s2 = 3
        for i in range(N):
            y = np.random.rand(1)
            # 每次采样需要迭代 k 次
            for j in range(k):
                x = pXgivenY(y, m1, m2, s1, s2)
                y = pYgivenX(x, m1, m2, s1, s2)
            x0[i] = x

        return x0

    def f(x):
        """目标分布"""
        return np.exp(-(x - 10) ** 2 / 10)

    # 画图
    N = 10000
    s = gibbs(N)
    x1 = np.arange(0, 17, 1)
    plt.hist(s, bins=x1, fc='c')
    x1 = np.arange(0, 17, 0.1)
    px1 = np.zeros(len(x1))
    for i in range(len(x1)):
        px1[i] = f(x1[i])
    plt.plot(x1, px1 * N * 10 / sum(px1), color='r', linewidth=3)

    plt.show()


if __name__ == '__main__':
    m()
