from scipy import stats as st
from matplotlib import pyplot as plt
import seaborn as sns
import math
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from prettytable import PrettyTable

names = ["Normal distribution", "Cauchy distribution", "Laplace distribution", "Poisson distribution",
         "Uniform distribution"]


class Distribution:
    def __init__(self, name=None, size=0, repeat_num=1000):
        self.a = None
        self.b = None
        self.name = name
        self.size = size
        self.random_numbers = None
        self.density = None
        self.pdf = None
        self.cdf = None
        self.repeat_num = repeat_num
        self.x = None

    def __repr__(self):
        return f"{self.name}\nOn interval: [{self.a}, {self.b}]\nSize: {self.size}\nRandom numbers: " \
               f"{self.random_numbers}\nDensity: {self.density}\n\n"

    def set_distribution(self):
        if self.name == names[0]:
            self.random_numbers = st.norm.rvs(size=self.size)
            self.density = st.norm()
        elif self.name == names[1]:
            self.random_numbers = st.cauchy.rvs(size=self.size)
            self.density = st.cauchy()
        elif self.name == names[2]:
            self.random_numbers = st.laplace.rvs(size=self.size)
            self.density = st.laplace(scale=1 / math.sqrt(2), loc=0)
        elif self.name == names[3]:
            self.random_numbers = st.poisson.rvs(mu=10, size=self.size)
            self.density = st.poisson(10)  # mu = 10
        elif self.name == names[4]:
            a = -math.sqrt(3)
            step = 2 * math.sqrt(3)
            self.random_numbers = st.uniform.rvs(size=self.size, loc=a, scale=step)
            self.density = st.uniform(loc=a, scale=step)

    def set_x_pdf(self):
        if self.name == names[3]:
            self.x = np.arange(self.density.ppf(0.01), self.density.ppf(0.99))
            self.pdf = self.density.pmf(self.x)
        else:
            self.x = np.linspace(self.density.ppf(0.01), self.density.ppf(0.99), num=100)
            self.pdf = self.density.pdf(self.x)

    def set_x_cdf_pdf(self, param: str):
        self.x = np.linspace(self.a, self.b, self.repeat_num)
        if self.name == names[0]:
            self.pdf = st.norm.pdf(self.x)
            self.cdf = st.norm.cdf(self.x)
        elif self.name == names[1]:
            self.pdf = st.cauchy.pdf(self.x)
            self.cdf = st.cauchy.cdf(self.x)
        elif self.name == names[2]:
            self.pdf = st.laplace.pdf(self.x, loc=0, scale=1 / math.sqrt(2))
            self.cdf = st.laplace.cdf(self.x, loc=0, scale=1 / math.sqrt(2))
        elif self.name == names[3]:
            if param == 'kde':
                self.x = np.linspace(self.a, self.b, -self.a + self.b + 1)
            self.pdf = st.poisson(10).pmf(self.x)
            self.cdf = st.poisson(10).cdf(self.x)
        elif self.name == names[4]:
            a = -math.sqrt(3)
            step = 2 * math.sqrt(3)
            self.x = np.linspace(self.a, self.b, self.repeat_num)
            self.pdf = st.uniform.pdf(self.x, loc=a, scale=step)
            self.cdf = st.uniform.cdf(self.x, loc=a, scale=step)

    def set_a_b(self, a, b, a_poisson, b_poisson):
        if self.name == names[3]:
            self.a, self.b = a_poisson, b_poisson
        else:
            self.a, self.b = a, b
            
colors = ["deepskyblue", "limegreen", "tomato", "blueviolet", "orange"]


def build_histogram(dist_names, sizes):
    labels = ["size", "distribution"]
    line_type = "k--"
    for i, dist_name in enumerate(dist_names):
        for size in sizes:
            dist = Distribution(dist_name, size)
            dist.set_distribution()
            fig, ax = plt.subplots(1, 1)
            ax.hist(dist.random_numbers, density=True, alpha=0.7, histtype='stepfilled', color=colors[i])
            dist.set_x_pdf()
            ax.plot(dist.x, dist.pdf, line_type)
            ax.set_xlabel(labels[0] + ": " + str(size))
            ax.set_ylabel(labels[1])
            ax.set_title(dist_name)
            plt.grid()
            plt.show()
          
repeat_num = 1000


def calc_characteristics(dist_names, sizes):
    for dist_name in dist_names:
        for size in sizes:
            mean_list, median_list, z_r_list, z_q_list, z_tr_list, e_list, d_list, e_plus_minus_sqrt_d = [], [], [], [], [], [], [], []
            lists = [mean_list, median_list, z_r_list, z_q_list, z_tr_list]
            for i in range(repeat_num):
                dist = Distribution(dist_name, size)
                dist.set_distribution()
                arr = sorted(dist.random_numbers)
                mean_list.append(np.mean(arr))
                median_list.append(np.median(arr))
                z_r_list.append(z_r(arr, size))
                z_q_list.append(z_q(arr, size))
                z_tr_list.append(z_tr(arr, size))
            for elem in lists:
                e_list.append(round(np.mean(elem), 6))
                d_list.append(round(np.std(elem) ** 2, 6))
                e_plus_minus_sqrt_d.append([round(round(np.mean(elem), 6) - math.sqrt(round(np.std(elem) ** 2, 6)), 6),
                                            round(round(np.mean(elem), 6) + math.sqrt(round(np.std(elem) ** 2, 6)), 6)])
            table = PrettyTable()
            table.field_names = [f"{dist_name} n = " + str(size), "Mean", "Median", "Zr", "Zq", "Ztr"]
            e_list.insert(0, 'E(z)')
            d_list.insert(0, 'D(z)')
            e_plus_minus_sqrt_d.insert(0, 'E(z) +- sqrtD(z)')
            table.add_row(e_list)
            table.add_row(d_list)
            table.add_row(e_plus_minus_sqrt_d)
            # print(e_plus_minus_sqrt_d)
            print(table)


def z_r(selection, size):
    return (selection[0] + selection[size - 1]) / 2


def z_p(selection, n_p):
    if n_p.is_integer():
        return selection[int(n_p)]
    else:
        return selection[int(n_p) + 1]


def z_q(selection, size):
    return (z_p(selection, size / 4) + z_p(selection, 3 * size / 4)) / 2


def z_tr(selection, size):
    r = int(size / 4)
    amount = 0
    for i in range(r + 1, size - r + 1):
        amount += selection[i]
    return (1 / (size - 2 * r)) * amount
def build_boxplot(dist_names, sizes):
    for dist_name in dist_names:
        tips = []
        for size in sizes:
            dist = Distribution(dist_name, size)
            emission_share(dist, dist.repeat_num)
            tips.append(dist.random_numbers)
        draw_boxplot(dist_name, tips)


def mustache(distribution):
    q_1, q_3 = np.quantile(distribution, [0.25, 0.75])
    return q_1 - 3 / 2 * (q_3 - q_1), q_3 + 3 / 2 * (q_3 - q_1)


def count_emissions(distribution):
    x1, x2 = mustache(distribution)
    filtered = [x for x in distribution if x > x2 or x < x1]
    return len(filtered)


def emission_share(dist, repeat_num):
    count = 0
    for i in range(repeat_num):
        dist.set_distribution()
        arr = sorted(dist.random_numbers)
        count += count_emissions(arr)
    count /= (dist.size * repeat_num)
    dist.set_distribution()
    print(f"{dist.name} Size {dist.size}: {count}")


def draw_boxplot(dist_name, data):
    sns.set_theme(style="whitegrid")
    sns.boxplot(data=data, palette='pastel', orient='h')
    sns.despine(offset=10)
    plt.xlabel("x")
    plt.ylabel("n")
    plt.title(dist_name)
    # plt.savefig(img_path)
    plt.show()
    
a, b = -4, 4
a_poisson, b_poisson = 6, 14
coefs = [0.5, 1, 2]


def draw_ecdf(dist_names, sizes):
    sns.set_style("whitegrid")
    for dist_name in dist_names:
        figures, axs = plt.subplots(ncols=3, figsize=(15, 5))
        for i, size in enumerate(sizes):
            dist, arr = prepare_to_draw(dist_name, size, 'ecdf')
            ecdf = ECDF(arr)
            axs[i].plot(dist.x, dist.cdf, color="blue", label="cdf")
            axs[i].plot(dist.x, ecdf(dist.x), color="red", label="ecdf")
            axs[i].legend(loc='lower right')
            axs[i].set(xlabel="x", ylabel="F(x)")
            axs[i].set_title(f"n = {size}")
        figures.suptitle(dist_name)
        plt.show()


def draw_kde(dist_names, sizes):
    sns.set_style("whitegrid")
    for dist_name in dist_names:
        for size in sizes:
            figures, axs = plt.subplots(ncols=3, figsize=(15, 5))
            dist, arr = prepare_to_draw(dist_name, size, 'kde')
            for i, coef in enumerate(coefs):
                axs[i].plot(dist.x, dist.pdf, color="red", label="pdf")
                sns.kdeplot(data=arr, bw_method="silverman", bw_adjust=coef, ax=axs[i], fill=True, linewidth=0, label="kde")
                axs[i].legend(loc="upper right")
                axs[i].set(xlabel="x", ylabel="f(x)")
                axs[i].set_xlim([dist.a, dist.b])
                axs[i].set_title("h = " + str(coef))
            figures.suptitle(f"{dist_name} KDE n = {size}")
            plt.show()


def prepare_to_draw(dist_name, size, param):
    dist = Distribution(dist_name, size)
    dist.set_a_b(a, b, a_poisson, b_poisson)
    dist.set_distribution()
    dist.set_x_cdf_pdf(param)
    arr = sorted(dist.random_numbers)
    return dist, arr

if __name__ == "__main__":
    # initial conditions
    sizes = [[10, 50, 1000], [10, 100, 1000], [20, 100], [20, 60, 100]]

    # LABS
    build_histogram(names, sizes[0])  # lab 1
    calc_characteristics(names, sizes[1])  # lab 2
    build_boxplot(names, sizes[2])  # lab 3
    draw_ecdf(names, sizes[3])  # lab 4.1
    draw_kde(names, sizes[3])  # lab 4.2
  
