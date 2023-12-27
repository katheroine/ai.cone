# Source: https://medium.com/analytics-vidhya/how-to-draw-a-3d-christmas-tree-with-matplotlib-aabb9bc27864

import math
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection="3d")


def init():
    k=300
    Z = [i for i in range(k)]
    X = [math.cos(i/5)*(k-i) for i in range(k)]
    Y = [math.sin(i/5)*(k-i) for i in range(k)]
    ax.scatter(X,Y,Z, c="green", marker="^")
    step = 3
    c = [(i/k,abs(0.5-i/k),i/k) for i in range(1,k,step)]
    Z = [i for i in range(1,k,step)]
    X = [math.cos(i/5+2)*(k-i+10) for i in range(1,k,step)]
    Y = [math.sin(i/5+2)*(k-i+10) for i in range(1,k,step)]
    ax.scatter(X,Y,Z, c=c, marker="o",s=40)
    plt.xlim(-500,500)
    plt.ylim(-500,500)
    return fig,


def animate(f):
    fig.clear()
    ax = fig.add_subplot(111, projection="3d")
    k=300
    Z = [i for i in range(k)]
    X = [math.cos(i/5+f/10)*(k-i) for i in range(k)]
    Y = [math.sin(i/5+f/10)*(k-i) for i in range(k)]
    ax.scatter(X,Y,Z, c="green", marker="^")
    step = 3
    c = [(i/k,abs(0.5-i/k),i/k) for i in range(1,k,step)]
    Z = [i for i in range(1,k,step)]
    X = [math.cos(i/5+2+f/10)*(k-i+10) for i in range(1,k,step)]
    Y = [math.sin(i/5+2+f/10)*(k-i+10) for i in range(1,k,step)]
    ax.scatter(X,Y,Z, c=c, marker="o",s=40)
    plt.xlim(-500,500)
    plt.ylim(-500,500)
    return fig,

ani=animation.FuncAnimation(fig, animate, init_func=init,
                               frames=90, interval=50, blit=True)

ani.save("xmas_tree.mp4")