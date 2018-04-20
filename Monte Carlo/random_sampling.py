# coding: utf-8
from pylab import *

"""采样问题：分布->数据
计算机可以获得服从均匀分布的随机数（容易根据种子生成伪随机数）。
一般的采样问题，都可以理解成：有了均匀分布（简单分布）的采样，如何去获取复杂分布的采样。
"""

"""离散随机采样
"""


def d():
    index2word = ["你", "好", "合", "协"]

    def sample_discrete(vec):
        u = np.random.rand()
        start = 0
        for i, num in enumerate(vec):
            if u > start:
                start += num
            else:
                return i - 1
        return i

    count = dict([(w, 0) for w in index2word])
    # 采样1000次
    for i in range(1000):
        s = sample_discrete([0.1, 0.5, 0.2, 0.2])
        count[index2word[s]] += 1
    for k in count:
        print k, " : ", count[k]


"""连续随机采样
对于连续的分布，如果可以计算这个分布的累积分布函数（CDF），就可以通过计算CDF的反函数，结合基础的均匀分布，获得其采样。
Box-Muller算法：利用均匀分布实现对高斯分布的采样。
"""


def c():
    def boxmuller(n):

        x = np.zeros((n, 2))
        y = np.zeros((n, 2))

        for i in range(n):
            x[i, :] = np.array([2, 2])
            x2 = x[i, 0] * x[i, 0] + x[i, 1] * x[i, 1]
            while (x2) > 1:
                x[i, :] = np.random.rand(2) * 2 - 1
                x2 = x[i, 0] * x[i, 0] + x[i, 1] * x[i, 1]

            y[i, :] = x[i, :] * np.sqrt((-2 * log(x2)) / x2)

        y = np.reshape(y, 2 * n, 1)
        return y

    y = boxmuller(1000)
    hist(y, normed=1, fc='c')
    x = arange(-4, 4, 0.1)
    plot(x, 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x ** 2), 'g', lw=6)
    xlabel('x', fontsize=24)
    ylabel('p(x)', fontsize=24)
    show()


"""接受-拒绝采样
"""


def r():
    def qsample():
        """使用均匀分布作为q(x)，返回采样"""
        return np.random.rand() * 4.

    def p(x):
        """目标分布"""
        return 0.3 * np.exp(-(x - 0.3) ** 2) + 0.7 * np.exp(-(x - 2.) ** 2 / 0.3)

    def rejection(nsamples):

        M = 0.72  # 0.8 k值
        samples = np.zeros(nsamples, dtype=float)
        count = 0
        for i in range(nsamples):
            accept = False
            while not accept:
                x = qsample()
                u = np.random.rand() * M
                if u < p(x):
                    accept = True
                    samples[i] = x
                else:
                    count += 1
        print "reject count: ", count
        return samples

    x = np.arange(0, 4, 0.01)
    x2 = np.arange(-0.5, 4.5, 0.1)
    realdata = 0.3 * np.exp(-(x - 0.3) ** 2) + 0.7 * np.exp(-(x - 2.) ** 2 / 0.3)
    box = np.ones(len(x2)) * 0.75  # 0.8
    box[:5] = 0
    box[-5:] = 0
    plt.plot(x, realdata, 'g', lw=3)
    plt.plot(x2, box, 'r--', lw=3)

    import time
    t0 = time.time()
    samples = rejection(10000)
    t1 = time.time()
    print "Time ", t1 - t0

    plt.hist(samples, 15, normed=1, fc='c')
    plt.xlabel('x', fontsize=24)
    plt.ylabel('p(x)', fontsize=24)
    plt.axis([-0.5, 4.5, 0, 1])
    plt.show()


"""重要性采样
"""


def i():
    def qsample():
        return np.random.rand() * 4.

    def p(x):
        return 0.3 * np.exp(-(x - 0.3) ** 2) + 0.7 * np.exp(-(x - 2.) ** 2 / 0.3)

    def q(x):
        return 4.0

    def importance(nsamples):

        samples = np.zeros(nsamples, dtype=float)
        w = np.zeros(nsamples, dtype=float)

        for i in range(nsamples):
            samples[i] = qsample()
            w[i] = p(samples[i]) / q(samples[i])

        return samples, w

    def sample_discrete(vec):
        u = np.random.rand()
        start = 0
        for i, num in enumerate(vec):
            if u > start:
                start += num
            else:
                return i - 1
        return i

    def importance_sampling(nsamples):
        samples, w = importance(nsamples)
        #     print samples
        final_samples = np.zeros(nsamples, dtype=float)
        w = w / w.sum()
        #     print w
        for j in range(nsamples):
            final_samples[j] = samples[sample_discrete(w)]
        return final_samples

    x = np.arange(0, 4, 0.01)
    x2 = np.arange(-0.5, 4.5, 0.1)
    realdata = 0.3 * np.exp(-(x - 0.3) ** 2) + 0.7 * np.exp(-(x - 2.) ** 2 / 0.3)
    box = np.ones(len(x2)) * 0.8
    box[:5] = 0
    box[-5:] = 0
    plt.plot(x, realdata, 'g', lw=6)
    plt.plot(x2, box, 'r--', lw=6)

    # samples,w = importance(5000)
    final_samples = importance_sampling(5000)
    plt.hist(final_samples, normed=1, fc='c')
    plt.show()


if __name__ == '__main__':
    c()
