import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

########### ここからを変更 ##############
WIDTH = 1640
HEIGHT = 1232

mtx = np.array([[1.69806089e+03, 0.00000000e+00, 7.88050060e+02],
                [0.00000000e+00, 1.70364174e+03, 6.38980510e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.array([0.71281358, -0.72957719,  0.01985307, -0.01629121, -0.4710592])
########### ここまでを変更 ##############

k1 = dist[0]; k2 = dist[1]; k3 = dist[4]
p1 = dist[2]; p2 = dist[3]
fx = mtx[0][0]; fy = mtx[1][1]
cx = mtx[0][2]; cy = mtx[1][2]
ifx = 1./fx; ify = 1./fy

def calc_distortion(x, y):
    # カメラ座標系での点
    x = (x - cx)*ifx
    y = (y - cy)*ify

    r2 = x*x + y*y
    r4 = r2*r2
    r6 = r4*r2
    a1 = 2*x*y
    a2 = r2 + 2*x*x
    a3 = r2 + 2*y*y

    xd = x * (1 + k1*r2 + k2*r4 + k3*r6) + p1*a1 + p2*a2
    yd = y * (1 + k1*r2 + k2*r4 + k3*r6) + p1*a3 + p2*a1

    x_proj = xd*fx + cx
    y_proj = yd*fy + cy

    return x_proj, y_proj

def make_equations(xp, yp):
    def equations(vars):
        xd, yd = vars
        r2 = xd*xd + yd*yd
        r4 = r2*r2
        r6 = r4*r2
        a1 = 2*xd*yd
        a2 = r2 + 2*xd*xd
        a3 = r2 + 2*yd*yd

        eq1 = xd * (1 + k1*r2 + k2*r4 + k3*r6) + p1*a1 + p2*a2 - xp
        eq2 = yd * (1 + k1*r2 + k2*r4 + k3*r6) + p1*a3 + p2*a1 - yp
        return [eq1, eq2]
    return equations



def main():
    ox = []; oy = [] # 元のx[px], y[px]
    x = []; y = [] # 歪計算後のx[px], y[px]

    # 20 pxおきに計算
    for i in range(10, WIDTH-10, 20):
        for j in range(10, HEIGHT-10, 20):
            xp, yp = calc_distortion(i, j) # 歪計算
            x.append(xp)
            y.append(yp)
            ox.append(i)
            oy.append(j)

    ox2 = []; oy2 = [] # 歪を直した後のx[px], y[px]

    # 非線形方程式を解き、歪を直す
    for i in range(len(x)):
        xp = (x[i] - cx)*ifx
        yp = (y[i] - cy)*ify

        initial_val = [xp, yp]
        solution = root(make_equations(xp, yp), initial_val, method='hybr')

        xd, yd = solution.x
        xdp = xd*fx + cx; ydp = yd*fy + cy
        ox2.append(xdp); oy2.append(ydp)
    
    # プロット
    fig, axes = plt.subplots(1, 3)
    axes[0].scatter(ox, oy, s=0.5)
    axes[0].set_aspect('equal')
    axes[0].set_title("Original x and y")
    axes[1].scatter(x, y, s=0.5)
    axes[1].set_aspect('equal')
    axes[1].set_title("After distortion")
    axes[2].scatter(ox2, oy2, s=0.5)
    axes[2].set_aspect('equal')
    axes[2].set_title("After distortion correction")
    plt.tight_layout()
    plt.show()

    # 歪修正の精度
    oxn = np.array(ox); oyn = np.array(oy)
    ox2n = np.array(ox2); oy2n = np.array(oy2)
    errx = oxn - ox2n; erry = oyn - oy2n
    err = np.sqrt(errx*errx + erry*erry)
    print(f"max_error = {np.max(err)}")

if __name__ == "__main__":
    main()
