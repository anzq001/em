import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号


def generateData(alphaArray, muArray, sigmaArray,dataNum):
    ratioArray = alphaArray/alphaArray.sum()
    idx = np.arange(0, len(ratioArray))
    data_idx = np.random.choice(idx, size=dataNum, p=ratioArray)
    dataArray = np.random.normal(muArray[data_idx], sigmaArray[data_idx], size=dataNum)
    return dataArray


def pdf(x, mu, sigma):
    return (1.0/(np.sqrt(2*np.pi)*sigma))*np.exp(-(x-mu)**2/(2*sigma**2))

def em(data, class_num):
    """
    :param data: 待预测的点
    :param class_num: 待预测的点的类型数
    :return: 预测得到的高斯模型参数
    """
    dataNum = data.__len__()
    alphaArray = np.ones(shape=[class_num], dtype=float)/class_num # 预设值数据量均分
    muArray = np.random.randn(class_num)
    sigmaArray = np.ones(shape=[class_num], dtype=float)
    gamaArray = np.zeros(shape=[dataNum, class_num])
    maxIter = 100
    epsilon = 0.001 # 设置停止条件为alpha, mu, sigma参数的变化不超过epsilon
    for step in np.arange(maxIter):
        for i in np.arange(dataNum):
            ratioArray = alphaArray * pdf(data[i], muArray, sigmaArray)
            gamaArray[i] = ratioArray/ratioArray.sum() # E step

        old_alphaArray = alphaArray
        old_muArray = muArray
        old_sigmaArray = sigmaArray

        alphaArray = np.sum(gamaArray, axis=0)/dataNum # M step
        muArray = data.T.dot(gamaArray)/np.sum(gamaArray, axis=0)
        for k in np.arange(class_num):
            sigmaArray[k] = np.sqrt((gamaArray[:, k].T.dot((data - muArray[k])**2))/np.sum(gamaArray[:, k]))
        # sigmaArray = np.sqrt(np.sum(gamaArray))

        alpha_diff = alphaArray - old_alphaArray
        mu_diff = muArray - old_muArray
        sigma_diff = sigmaArray - old_sigmaArray
        if alpha_diff.dot(alpha_diff.T) < epsilon and mu_diff.dot(mu_diff.T) < epsilon and sigma_diff.dot(sigma_diff.T) < epsilon:
            break
        print(alphaArray, muArray, sigmaArray)
    return alphaArray, muArray, sigmaArray


def draw_plot(alphaArray, muArray, sigmaArray, color=None, tag=None):
    ratioArray = alphaArray/alphaArray.sum()
    x_min = muArray - 3 * sigmaArray
    x_max = muArray + 3 * sigmaArray
    x_bottom = x_min.min()
    x_top = x_max.max()
    x_array = np.linspace(x_bottom, x_top, 200)
    y_array = np.zeros(200, dtype=float)
    for ratio, mu, sigma in zip(ratioArray, muArray, sigmaArray):
        y_array = y_array + ratio * pdf(x_array, mu, sigma)
    plt.figure(0)
    plt.plot(x_array, y_array, color=color, label=tag)
    plt.legend()
    plt.xlim([x_array.min(), x_array.max()])
    # plt.ylim([y_array.min(), y_array.max()])

def draw_hist(data):
    plt.figure(0)
    plt.hist(data, bins=200, density=True)
    plt.title("频数分布直方图")
    plt.xlabel("区间")
    plt.ylabel("频数")


if __name__ == '__main__':
    # 参数的准确值
    class_num = 2
    num = 10000
    alphaArray = np.array([0.2, 0.8])
    muArray = np.array([-2, 2])
    sigmaArray = np.array([1, 1])
    draw_plot(alphaArray, muArray, sigmaArray, color="r", tag="real")
    data = generateData(alphaArray, muArray, sigmaArray, num)
    draw_hist(data)

    # 使用em算法估计参数
    alpha_hat_array,mu_hat_array,sigma_hat_array = em(data, class_num)
    print(alpha_hat_array, mu_hat_array, sigma_hat_array)
    draw_plot(alpha_hat_array, mu_hat_array, sigma_hat_array, color="g", tag="predict")
    plt.show()

    test = -1
    ratio = alpha_hat_array * pdf(test, mu_hat_array, sigma_hat_array) #当值为test时，该值为各类的比例
    ratio = ratio/ratio.sum()
    tag = np.argmax(ratio)
    print("当前点属于N(%.3f, %.3f)类的概率为%.3f" % (mu_hat_array[tag], sigma_hat_array[tag], ratio[tag]))
