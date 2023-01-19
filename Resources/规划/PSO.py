#基本粒子群算法
#vi+1 = w*vi+c1*r1*(pi-xi)+c2*r2*(pg-xi)   速度更新公式
#xi+1 = xi + a*vi+1    位置更新公式（一般a=1）
#w = wmax -(wmax-wmin)*iter/Iter  权重更新公式
#iter当前迭代次数 Iter最大迭代次数 c1、c2学习因子  r1、r2随机数 pi粒子当前最优位置  pg粒子群全局最优
#初始化 wmax=0.9 wmin=0.4 通常c1=c2=2 Iter对于小规模问题（10，20）对于大规模（100，200）
#算法优劣取决于w、c1和c2，迭代结束的条件是适应度函数的值符合具体问题的要求
#初始化粒子群，包括尺寸、速度和位置
#本算法假设想要的输出是长度为10的矩阵，y=[1.7]*10,适应度函数f（x）= |x-y| <=0.001符合要求
#https://www.cnblogs.com/lemon-567/p/14377134.html
import numpy as np

swarmsize = 500
partlen = 10
wmax,wmin = 0.9,0.4
c1 = c2 = 2
Iter = 400

def getwgh(iter):
    w = wmax - (wmax-wmin)*iter/Iter
    return w

def getrange():
    randompv = (np.random.rand()-0.5)*2
    return randompv

def initswarm():
    vswarm,pswarm = np.zeros((swarmsize,partlen)),np.zeros((swarmsize,partlen))
    for i in range(swarmsize):
        for j in range(partlen):
            vswarm[i][j] = getrange()
            pswarm[i][j] = getrange()
    return vswarm,pswarm

def getfitness(pswarm):
    pbest = np.zeros(partlen)
    fitness = np.zeros(swarmsize)
    for i in range(partlen):
        pbest[i] = 1.7

    for i in range(swarmsize):
        yloss = pswarm[i] - pbest
        for j in range(partlen):
            fitness[i] += abs(yloss[j])
    return fitness

def getpgfit(fitness,pswarm):
    pgfitness = fitness.min()
    pg = pswarm[fitness.argmin()].copy()
    return pg,pgfitness

vswarm,pswarm = initswarm()
fitness = getfitness(pswarm)
pg,pgfit = getpgfit(fitness,pswarm)
pi,pifit = pswarm.copy(),fitness.copy()

for iter in range(Iter):
    if pgfit <= 0.001:
        break
    #更新速度和位置
    weight = getwgh(iter)
    for i in range(swarmsize):
        for j in range(partlen):
            vswarm[i][j] = weight*vswarm[i][j] + c1*np.random.rand()*(pi[i][j]-pswarm[i][j]) + c2*np.random.rand()*(pg[j]-pswarm[i][j])
            pswarm[i][j] = pswarm[i][j] + vswarm[i][j]
    #更新适应值
    fitness = getfitness(pswarm)
    #更新全局最优粒子
    pg,pgfit = getpgfit(fitness,pswarm)
    #更新局部最优粒子
    for i in range(swarmsize):
        if fitness[i] < pifit[i]:
            pifit[i] = fitness[i].copy()
            pi[i] = pswarm[i].copy()

for j in range(swarmsize):
    if pifit[j] < pgfit:
        pgfit = pifit[j].copy()
        pg = pi[j].copy()
print(pg)
print(pgfit)