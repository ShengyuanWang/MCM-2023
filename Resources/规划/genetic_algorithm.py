import numpy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# https://blog.csdn.net/ha_ha_ha233/article/details/91364937
DNA_SIZE = 24
POP_SIZE = 2000
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.005
N_GENERATIONS = 200
X_BOUND = [0, 8]
Y_BOUND = [0, 7.5]


# 目标函数
def F(x, y):
    z = x
    if z.size != 1:
        return z
    for i in range(POP_SIZE):
        if 6 * x[i] + 5 * y[i] <= 50 and 10 * x[i] + 20 * y[i] <= 150 and x[i] >= 0 and y[i] >= 0 and x[i] <= 8:
            z[i] = 10 * x[i] + 9 * y[i]
        else:
            z[i] = -1000000
    return z
    # z = x
    # return 10 * x + 9 * y


# 绘图，按照坐标展示函数的值
def plot_3d(ax):
    X = np.linspace(*X_BOUND, 100)
    Y = np.linspace(*Y_BOUND, 100)
    X, Y = np.meshgrid(X, Y)
    Z = F(X, Y)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)
    ax.set_zlim(-10, 10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.pause(3)
    plt.show()


# 求适应度
def get_fitness(pop):
    x, y = translateDNA(pop)
    pred = F(x, y)
    return (pred - np.min(
        pred)) + 1e-3  # 减去最小的适应度是为了防止适应度出现负数，通过这一步fitness的范围为[0, np.max(pred)-np.min(pred)],最后在加上一个很小的数防止出现为0的适应度


# 将x, y编码到同一个48位的二进制串里面，每一个变量用24位表示，奇数24位——x; 偶数24位——y;
def translateDNA(pop): # pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
    x_pop = pop[:, 1::2]  # 奇数列表示X
    y_pop = pop[:, ::2]  # 偶数列表示y

    # pop:(POP_SIZE,DNA_SIZE)*(DNA_SIZE,1) --> (POP_SIZE,1)
    x = x_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
    y = y_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (Y_BOUND[1] - Y_BOUND[0]) + Y_BOUND[0]
    return x, y


# 交叉和变异
def crossover_and_mutation(pop, CROSSOVER_RATE=0.8):
    new_pop = []
    for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲
        child = father  # 孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
        if np.random.rand() < CROSSOVER_RATE:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
            mother = pop[np.random.randint(POP_SIZE)]  # 再种群中选择另一个个体，并将该个体作为母亲
            cross_points = np.random.randint(low=0, high=DNA_SIZE * 2)  # 随机产生交叉的点
            child[cross_points:] = mother[cross_points:]  # 孩子得到位于交叉点后的母亲的基因
        mutation(child)  # 每个后代有一定的机率发生变异
        new_pop.append(child)

    return new_pop


def mutation(child, MUTATION_RATE=0.003):
    if np.random.rand() < MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
        mutate_point = np.random.randint(0, DNA_SIZE * 2)  # 随机产生一个实数，代表要变异基因的位置
        child[mutate_point] = child[mutate_point] ^ 1  # 将变异点的二进制为反转


# 按照适应度高低进行选择
def select(pop, fitness):  # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=(fitness) / (fitness.sum()))
    return pop[idx]


def print_info(pop):
    fitness = get_fitness(pop)
    max_fitness_index = np.argmax(fitness)
    print("max_fitness:", fitness[max_fitness_index])
    x, y = translateDNA(pop)
    print("最优的基因型：", pop[max_fitness_index])
    print("(x, y):", (x[max_fitness_index], y[max_fitness_index]))


if __name__ == "__main__":
    fig = plt.figure()
    ax = Axes3D(fig)
    plt.ion()  # 将画图模式改为交互模式，程序遇到plt.show不会暂停，而是继续执行
    plot_3d(ax)

    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE * 2))  # matrix (POP_SIZE, DNA_SIZE)
    for _ in range(N_GENERATIONS):  # 迭代N代
        x, y = translateDNA(pop)
        if 'sca' in locals():
            sca.remove()
        sca = ax.scatter(x, y, F(x, y), c='black', marker='o');
        plt.show();
        plt.pause(0.1)
        pop = np.array(crossover_and_mutation(pop, CROSSOVER_RATE))
        # F_values = F(translateDNA(pop)[0], translateDNA(pop)[1])#x, y --> Z matrix
        fitness = get_fitness(pop)
        pop = select(pop, fitness)  # 选择生成新的种群

    print_info(pop)
    plt.ioff()
    plot_3d(ax)
