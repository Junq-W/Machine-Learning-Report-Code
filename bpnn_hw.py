# 定义一个三个输入以上的逻辑问题：例如A AND B OR C
import numpy as np
def act(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# 10个隐藏层节点
NH = 10
# 定义输入，维度为8*3
x = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 1]])
# 对应的输出值，维度为8*1
y = np.array([[0], [1], [0], [0], [1], [1], [1], [1]])

# 为了将输出映射到[-1,1]之间。注意维度：w1维度是3*10，w2维度是10*1
w1 = 2 * np.random.random((3, NH)) - 1
w2 = 2 * np.random.random((NH, 1)) - 1

# 前向传播：
def feedfoward(x):
    a0 = x   # a0是输入维度是8*3
    z1 = np.dot(a0, w1)
    a1 = act(z1)    # a1是第一层的结果，维度是8*3*3*10 --> 8*10
    z2 = np.dot(a1, w2)
    a2 = act(z2)    # a2是第二次的结果。维度是8*10*10*1 --> 8*1
    return (z1,z2),(a0, a1, a2)

n_epochs = 100000
for i in range(n_epochs):
    (z1,z2),(a0, a1, a2) = feedfoward(x)
    # print(z1,a1)
    # print(z2,a2)
    l2_delta = (a2 - y) * act(a2, deriv=True)
    l1_delta = l2_delta.dot(w2.T) * act(a1, deriv=True)
    w2 = w2 - a1.T.dot(l2_delta) * 0.1
    w1 = w1 - a0.T.dot(l1_delta) * 0.1
    # print(w1)
    if (i % 10000) == 0:
        loss = np.mean(np.square(y - a2))
        print("epochs %d/%d loss = %f" % (i / 1e4 + 1, n_epochs / 1e4, loss))

(z1,z2),(a0, a1, a2) = feedfoward(x)
print("A AND B OR C(", x, ") = ", a2)

