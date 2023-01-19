# mathmodel05_v1.py
# Demo05 of mathematical modeling algorithm
# Solving integer programming with PuLP.
# Copyright 2021 Youcans, XUPT
# Crated：2021-05-31
# Python小白的数学建模课 @ Youcans
# https://www.cnblogs.com/youcans/p/14844841.html
import pulp  # 导入 pulp 库
# cplex

# 主程序
def main():
    # 模型参数设置
    """
    问题描述：
        某厂生产甲乙两种饮料，每百箱甲饮料需用原料6千克、工人10名，获利10万元；每百箱乙饮料需用原料5千克、工人20名，获利9万元。
        今工厂共有原料60千克、工人150名，又由于其他条件所限甲饮料产量不超过8百箱。
        （1）问如何安排生产计划，即两种饮料各生产多少使获利最大？
        （2）若投资0.8万元可增加原料1千克，是否应作这项投资？投资多少合理？
        （3）若不允许散箱（按整百箱生产），如何安排生产计划，即两种饮料各生产多少使获利最大？
        （4）若不允许散箱（按整百箱生产），若投资0.8万元可增加原料1千克，是否应作这项投资？投资多少合理？
    """

    # 问题 1：
    """
    问题建模：
        决策变量：
            x1：甲饮料产量（单位：百箱）
            x2：乙饮料产量（单位：百箱）
        目标函数：
            max fx = 10*x1 + 9*x2
        约束条件：
            6*x1 + 5*x2 <= 60
            10*x1 + 20*x2 <= 150            
            x1, x2 >= 0，x1 <= 8
    此外，由 x1,x2>=0 和 10 * x1 + 20 * x2 <= 150 可知 0 <= x2 <= 7.5
    """
    ProbLP1 = pulp.LpProblem("ProbLP1", sense=pulp.LpMaximize)  # 定义问题 1，求最大值
    x1 = pulp.LpVariable('x1', lowBound=0, upBound=8, cat='Continue')  # 定义 x1
    x2 = pulp.LpVariable('x2', lowBound=0, upBound=7.5, cat='Continue')  # 定义 x2
    ProbLP1 += (10 * x1 + 9 * x2)  # 设置目标函数 f(x)
    ProbLP1 += (6 * x1 + 5 * x2 <= 60)  # 不等式约束
    ProbLP1 += (10 * x1 + 20 * x2 <= 150)  # 不等式约束
    ProbLP1.solve()
    print(ProbLP1.name)  # 输出求解状态
    print("Status youcans:", pulp.LpStatus[ProbLP1.status])  # 输出求解状态
    for v in ProbLP1.variables():
        print(v.name, "=", v.varValue)  # 输出每个变量的最优值
    print("F1(x) =", pulp.value(ProbLP1.objective))  # 输出最优解的目标函数值

    # 问题 2：
    """
    问题建模：
        决策变量：
            x1：甲饮料产量（单位：百箱）
            x2：乙饮料产量（单位：百箱）
            x3：增加投资（单位：万元）
        目标函数：
            max fx = 10*x1 + 9*x2 - x3
        约束条件：
            6*x1 + 5*x2 <= 60 + x3/0.8
            10*x1 + 20*x2 <= 150
            x1, x2, x3 >= 0，x1 <= 8
    此外，由 x1,x2>=0 和 10*x1+20*x2<=150 可知 0<=x2<=7.5
    """
    ProbLP2 = pulp.LpProblem("ProbLP2", sense=pulp.LpMaximize)  # 定义问题 2，求最大值
    x1 = pulp.LpVariable('x1', lowBound=0, upBound=8, cat='Integer')  # 定义 x1
    x2 = pulp.LpVariable('x2', lowBound=0, upBound=7.5, cat='Integer')  # 定义 x2
    x3 = pulp.LpVariable('x3', lowBound=0, cat='Integer')  # 定义 x3
    ProbLP2 += (10 * x1 + 9 * x2 - x3)  # 设置目标函数 f(x)
    ProbLP2 += (6 * x1 + 5 * x2 - 1.25 * x3 <= 60)  # 不等式约束
    ProbLP2 += (10 * x1 + 20 * x2 <= 150)  # 不等式约束
    ProbLP2.solve()
    print(ProbLP2.name)  # 输出求解状态
    print("Status  youcans:", pulp.LpStatus[ProbLP2.status])  # 输出求解状态
    for v in ProbLP2.variables():
        print(v.name, "=", v.varValue)  # 输出每个变量的最优值
    print("F2(x) =", pulp.value(ProbLP2.objective))  # 输出最优解的目标函数值

    # 问题 3：整数规划问题
    """
    问题建模：
        决策变量：
            x1：甲饮料产量，正整数（单位：百箱）
            x2：乙饮料产量，正整数（单位：百箱）
        目标函数：
            max fx = 10*x1 + 9*x2
        约束条件：
            6*x1 + 5*x2 <= 60
            10*x1 + 20*x2 <= 150
            x1, x2 >= 0，x1 <= 8，x1, x2 为整数
    此外，由 x1,x2>=0 和 10*x1+20*x2<=150 可知 0<=x2<=7.5
    """
    ProbLP3 = pulp.LpProblem("ProbLP3", sense=pulp.LpMaximize)  # 定义问题 3，求最大值
    print(ProbLP3.name)  # 输出求解状态
    x1 = pulp.LpVariable('x1', lowBound=0, upBound=8, cat='Integer')  # 定义 x1，变量类型：整数
    x2 = pulp.LpVariable('x2', lowBound=0, upBound=7.5, cat='Integer')  # 定义 x2，变量类型：整数
    ProbLP3 += (10 * x1 + 9 * x2)  # 设置目标函数 f(x)
    ProbLP3 += (6 * x1 + 5 * x2 <= 60)  # 不等式约束
    ProbLP3 += (10 * x1 + 20 * x2 <= 150)  # 不等式约束
    ProbLP3.solve()
    print("Shan Status:", pulp.LpStatus[ProbLP3.status])  # 输出求解状态
    for v in ProbLP3.variables():
        print(v.name, "=", v.varValue)  # 输出每个变量的最优值
    print("F3(x) =", pulp.value(ProbLP3.objective))  # 输出最优解的目标函数值

    # 问题 4：
    """
    问题建模：
        决策变量：
            x1：甲饮料产量，正整数（单位：百箱）
            x2：乙饮料产量，正整数（单位：百箱）
            x3：增加投资（单位：万元）
        目标函数：
            max fx = 10*x1 + 9*x2 - x3
        约束条件：
            6*x1 + 5*x2 <= 60 + x3/0.8
            10*x1 + 20*x2 <= 150
            x1, x2, x3 >= 0，x1 <= 8，x1, x2 为整数
    此外，由 x1,x2>=0 和 10*x1+20*x2<=150 可知 0<=x2<=7.5
    """
    ProbLP4 = pulp.LpProblem("ProbLP4", sense=pulp.LpMaximize)  # 定义问题 4，求最大值
    print(ProbLP4.name)  # 输出求解状态
    x1 = pulp.LpVariable('x1', lowBound=0, upBound=8, cat='Integer')  # 定义 x1，变量类型：整数
    x2 = pulp.LpVariable('x2', lowBound=0, upBound=7, cat='Integer')  # 定义 x2，变量类型：整数
    x3 = pulp.LpVariable('x3', lowBound=0, cat='Integer')  # 定义 x3
    ProbLP4 += (10 * x1 + 9 * x2 - x3)  # 设置目标函数 f(x)
    ProbLP4 += (6 * x1 + 5 * x2 - 1.25 * x3 <= 60)  # 不等式约束
    ProbLP4 += (10 * x1 + 20 * x2 <= 150)  # 不等式约束
    ProbLP4.solve()
    print("Shan Status:", pulp.LpStatus[ProbLP4.status])  # 输出求解状态
    for v in ProbLP4.variables():
        print(v.name, "=", v.varValue)  # 输出每个变量的最优值
    print("F4(x) =", pulp.value(ProbLP4.objective))  # 输出最优解的目标函数值

    return


if __name__ == '__main__':  # Copyright 2021 YouCans, XUPT
    main()  # Python小白的数学建模课 @ Youcans
