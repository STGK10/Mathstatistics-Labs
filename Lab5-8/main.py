import statistics
import math
import scipy
import tabulate
import scipy.stats as stats
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from matplotlib import transforms
from matplotlib.patches import Ellipse


class Task1:
    def __init__(self):
        self.sizes = [20, 60, 100]
        self.iterations = 1000
        self.rhos = [0, 0.5, 0.9]

    def multivar_normal(self, size, rho):
        return stats.multivariate_normal.rvs([0, 0], [[1.0, rho], [rho, 1.0]], size=size)

    def mixed_multivar_normal(self, size, rho):
        arr = 0.9 * stats.multivariate_normal.rvs([0, 0], [[1, 0.9], [0.9, 1]], size) + \
            0.1 * stats.multivariate_normal.rvs([0, 0], [[10, -0.9], [-0.9, 10]], size)
        return arr

    def quadrant_coef(self, x, y):
        x_med = np.median(x)
        y_med = np.median(y)
        n = [0, 0, 0, 0]

        for i in range(len(x)):
            if x[i] >= x_med and y[i] >= y_med:
                n[0] += 1
            elif x[i] < x_med and y[i] >= y_med:
                n[1] += 1
            elif x[i] < x_med:
                n[2] += 1
            else:
                n[3] += 1

        return (n[0] + n[2] - n[1] - n[3]) / len(x)

    def pprint(self, arr):
        st = "& "
        for a in arr:
            st += str(a)
            st += ' & '
            #print("& " + a, end=' ')
        return st

    def run(self):
        for size in self.sizes:
            for rho in self.rhos:
                mean, sq_mean, disp = self.generate_stats(self.multivar_normal, size, rho)
                print(f"Normal\t Size = {size}\t Rho = {rho}\t Mean = {self.pprint(mean)}\t Squares mean = {self.pprint(sq_mean)}\t Dispersion = {self.pprint(disp)}")

            mean, sq_mean, disp = self.generate_stats(self.mixed_multivar_normal, size, 0)
            print(f"Mixed\t Size = {size}\t Mean = {self.pprint(mean)}\t Squares mean = {self.pprint(sq_mean)}\t Dispersion = {self.pprint(disp)}")

            self.draw_ellipse(size)

    def generate_stats(self, distr_generator, size, rho):
        names = {"pearson": list(), "spearman": list(), "quadrant": list()}

        for i in range(self.iterations):
            multi_var = distr_generator(size, rho)
            x = multi_var[:, 0]
            y = multi_var[:, 1]

            names['pearson'].append(stats.pearsonr(x, y)[0])
            names['spearman'].append(stats.spearmanr(x, y)[0])
            names['quadrant'].append(self.quadrant_coef(x, y))

        mean = list()
        sq_mean = list()
        disp = list()
        for val in names.values():
            mean.append(np.median(val))
            sq_mean.append(np.median([val[k] ** 2 for k in range(self.iterations)]))
            disp.append(statistics.variance(val))

        return np.around(mean, decimals=4), np.around(sq_mean, decimals=4), np.around(disp, decimals=4)

    def build_ellipse(self, x, y, ax, n_std=3.0):
        cov = np.cov(x, y)
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

        rad_x = np.sqrt(1 + pearson)
        rad_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=rad_x * 2, height=rad_y * 2, facecolor='none', edgecolor='navy')
        scale_x = np.sqrt(cov[0, 0]) * 3.0
        mean_x = np.mean(x)
        scale_y = np.sqrt(cov[1, 1]) * 3.0
        mean_y = np.mean(y)

        transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)

    def draw_ellipse(self, size):
        fig, ax = plt.subplots(1, 3)
        titles = [f"rho = {rho}" for rho in self.rhos]

        for i in range(len(self.rhos)):
            sample = self.multivar_normal(size, self.rhos[i])
            x, y = sample[:, 0], sample[:, 1]
            self.build_ellipse(x, y, ax[i])
            ax[i].grid()
            ax[i].scatter(x, y, s=5)
            ax[i].set_title(titles[i])

        plt.suptitle(f"Size {size}")
        plt.show()
        
      
      
class Task2:
    def __init__(self):
        self.a = -1.8
        self.b = 2
        self.step = 0.2

    def ref_func(self, x):
        return 2 * x + 2

    def depend(self, x):
        return [self.ref_func(xi) + stats.norm.rvs(0, 1) for xi in x]

    def get_least_squares_params(self, x, y):
        beta_1 = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x * x) - np.mean(x) ** 2)
        beta_0 = np.mean(y) - beta_1 * np.mean(x)
        return beta_0, beta_1

    def least_module(self, parameters, x, y) -> float:
        alpha_0, alpha_1 = parameters
        return sum([abs(y[i] - alpha_0 - alpha_1 * x[i])
                    for i in range(len(x))])

    def get_least_module_params(self, x, y):
        beta_0, beta_1 = self.get_least_squares_params(x, y)
        result = scipy.optimize.minimize(self.least_module, [beta_0, beta_1], args=(x, y), method='SLSQP')
        return result.x[0], result.x[1]

    def least_squares_method(self, x, y):
        beta_0, beta_1 = self.get_least_squares_params(x, y)
        print(f"beta_0 = {beta_0}\t beta_1 = {beta_1}")
        return [beta_0 + beta_1 * p
                for p in x]

    def least_modules_method(self, x, y):
        alpha_0, alpha_1 = self.get_least_module_params(x, y)
        print(f"alpha_0 = {alpha_0}\t alpha_1 = {alpha_1}")
        return [alpha_0 + alpha_1 * p
                for p in x]

    def plot(self, x, y, name: str) -> None:
        y_mnk = self.least_squares_method(x, y)
        y_mnm = self.least_modules_method(x, y)

        mnk_dist = sum((self.ref_func(x)[i] - y_mnk[i]) ** 2 for i in range(len(y)))
        mnm_dist = sum(abs(self.ref_func(x)[i] - y_mnm[i]) for i in range(len(y)))
        print(f"MNK distance = {mnk_dist}\t MNM distance = {mnm_dist}")

        plt.plot(x, self.ref_func(x), color="red", label="Ideal")
        plt.plot(x, y_mnk, color="blue", label="MNK")
        plt.plot(x, y_mnm, color="green", label="MNM")
        plt.scatter(x, y, c="purple", label="Sample")
        plt.xlim([self.a, self.b])
        plt.grid()
        plt.legend()
        plt.title(name)
        plt.show()

    def run(self):
        x = np.arange(self.a, self.b, self.step)
        y = self.depend(x)
        self.plot(x, y, "Distribution")
        y[0] += 10
        y[-1] -= 10
        # y = self.depend(x)
        self.plot(x, y, "Distribution with perturbation ")
class Task3:
    def get_probability(self, distr, limits, size):
        p_arr = np.array([])
        n_arr = np.array([])

        for idx in range(-1, len(limits)):
            prev_cdf = 0 if idx == -1 else stats.norm.cdf(limits[idx])
            cur_cdf = 1 if idx == len(limits) - 1 else stats.norm.cdf(limits[idx + 1])
            p_arr = np.append(p_arr, cur_cdf - prev_cdf)

            if idx == -1:
                n_arr = np.append(n_arr, len(distr[distr <= limits[0]]))
            elif idx == len(limits) - 1:
                n_arr = np.append(n_arr, len(distr[distr >= limits[-1]]))
            else:
                n_arr = np.append(n_arr, len(distr[(distr <= limits[idx + 1]) & (distr >= limits[idx])]))
        result = np.divide(np.multiply((n_arr - size * p_arr), (n_arr - size * p_arr)), p_arr * size)
        return n_arr, p_arr, result

    def get_k(self, size):
        return math.ceil(1.72 * (size) ** (1 / 3))

    def create_table(self, n_arr, p_arr, result, size, limits):
        decimal = 4
        cols = ["i", "limits", "n_i", "p_i", "np_i", "n_i - np_i", "/frac{(n_i-np_i)^2}{np_i}"]
        rows = []
        for i in range(0, len(n_arr)):
            if i == 0:
                boarders = [-np.inf, np.around(limits[0], decimals=decimal)]
            elif i == len(n_arr) - 1:
                boarders = [np.around(limits[-1], decimals=decimal), 'inf']
            else:
                boarders = [np.around(limits[i - 1], decimals=decimal), np.around(limits[i], decimals=decimal)]

            rows.append(
                [i + 1, boarders, n_arr[i], np.around(p_arr[i], decimals=decimal),
                 np.around(p_arr[i] * size, decimals=decimal),
                 np.around(n_arr[i] - size * p_arr[i], decimals=decimal),
                 np.around(result[i], decimals=decimal)]
            )

        rows.append([len(n_arr) + 1, "-",
                     np.sum(n_arr),
                     np.around(np.sum(p_arr), decimals=4),
                     np.around(np.sum(p_arr * size), decimals=decimal),
                     -np.around(np.sum(n_arr - size * p_arr), decimals=decimal),
                     np.around(np.sum(result), decimals=decimal)])
        print(tabulate(rows, cols, tablefmt="latex"))

    def get_rvs(self, name, size):
        distr = None
        if name == 'norm':
            distr = np.random.normal(0, 1, size=size)
        elif name == 'laplace':
            distr = stats.laplace.rvs(size=size, scale=1 / math.sqrt(2), loc=0)
        elif name == 'uniform':
            distr = stats.uniform.rvs(size=size, loc=-math.sqrt(3), scale=2 * math.sqrt(3))
        return distr

    def calculate(self, distr, p, k):
        mu = np.mean(distr)
        sigma = np.std(distr)

        print('mu = ' + str(np.around(mu, decimals=2)))
        print('sigma = ' + str(np.around(sigma, decimals=2)))

        limits = np.linspace(-1.1, 1.1, num=k - 1)
        chi_2 = stats.chi2.ppf(p, k - 1)
        print('chi_2 = ' + str(chi_2))
        return limits

    def run(self):
        distr_names = {'norm': 100, 'laplace': 20, 'uniform': 20}
        alpha = 0.05
        p = 1 - alpha

        for item in distr_names.items():
            distr = self.get_rvs(item[0], size=item[1])
            k = self.get_k(item[1])
            limits = self.calculate(distr, p, k)
            n_arr, p_arr, result = self.get_probability(distr, limits, item[1])
            self.create_table(n_arr, p_arr, result, item[1], limits)

class Task4:
    def mean(self, data):
        return np.mean(data)

    def dispersion(self, sample):
        return self.mean(list(map(lambda x: x * x, sample))) - (self.mean(sample)) ** 2

    def normal(self, size):
        return np.random.standard_normal(size=size)

    def run(self):
        alpha = 0.05

        m_dict = {"norm": list(), "asymp": list()}
        s_dict = {"norm": list(), "asymp": list()}
        x_all = list()
        for n in [20, 100]:
            x = self.normal(n)
            x_all.append(x)
            m = self.mean(x)
            s = np.sqrt(self.dispersion(x))
            m_n = [m - s * (stats.t.ppf(1 - alpha / 2, n - 1)) / np.sqrt(n - 1),
                  m + s * (stats.t.ppf(1 - alpha / 2, n - 1)) / np.sqrt(n - 1)]
            s_n = [s * np.sqrt(n) / np.sqrt(stats.chi2.ppf(1 - alpha / 2, n - 1)),
                  s * np.sqrt(n) / np.sqrt(stats.chi2.ppf(alpha / 2, n - 1))]

            m_dict["norm"].append(m_n)
            s_dict["norm"].append(s_n)

            m_as = [m - stats.norm.ppf(1 - alpha / 2) / np.sqrt(n), m + stats.norm.ppf(1 - alpha / 2) / np.sqrt(n)]
            e = (sum(list(map(lambda el: (el - m) ** 4, x))) / n) / s ** 4 - 3
            s_as = [s / np.sqrt(1 + stats.norm.ppf(1 - alpha / 2) * np.sqrt((e + 2) / n)),
                    s / np.sqrt(1 - stats.norm.ppf(1 - alpha / 2) * np.sqrt((e + 2) / n))]

            m_dict["asymp"].append(m_as)
            s_dict["asymp"].append(s_as)

        for key in m_dict.keys():
            print(f"m {key} : {m_dict[key][0]},  {m_dict[key][1]}")
            print(f"sigma {key}: {s_dict[key][0]},  {s_dict[key][1]}")
            self.draw_result(x_all, m_dict[key], s_dict[key], key)

    def draw_result(self, x_set, m_all, s_all, name):
        fig, (ax1, ax2) = plt.subplots(1,2)

        ax1.set_ylim(0, 1)
        ax1.set_title(name)
        ax1.hist(x_set[0], density=True, histtype="stepfilled", label='n = 20')
        ax1.legend(loc='upper right')
        ax2.set_title(name)
        ax2.set_ylim(0, 1)
        ax2.hist(x_set[1], density=True, histtype="stepfilled", label='n = 100')
        ax2.legend(loc='upper right')
        fig.show()
        fig, (ax3, ax4) = plt.subplots(1,2)

        ax3.set_title(name + ' "m"')
        ax3.set_ylim(0.9, 1.4)
        ax3.plot(m_all[0], [1, 1], 'go-', label='n = 20')
        ax3.plot(m_all[1], [1.1, 1.1], 'mo-', label='n = 100')
        ax3.legend()

        ax4.set_title(name + ' "sigma"')
        ax4.set_ylim(0.9, 1.4)
        ax4.plot(s_all[0], [1, 1], 'go-', label='n = 20')
        ax4.plot(s_all[1], [1.1, 1.1], 'mo-', label='n = 100')
        ax4.legend()
        fig.show()
        
        
if __name__ == "__main__":
    task1 = Task1()
    task2 = Task2()
    task3 = Task3()
    task4 = Task4()
    task1.run()
    task2.run()
    task3.run()
    task4.run()
      
