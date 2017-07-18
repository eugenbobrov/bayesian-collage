import graph_cut
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import time

class GM():
    def __init__(self, unr, hor, ver, metric):
        """
        Конструктор, принимает на вход параметры энергии модели.

        Вход:
        unr -- массив N x M x K вещественных чисел, унарные потенциалы
        hor -- массив N x (M - 1) вещественных чисел, параметры парных
            потенциалов горизонтальных рёбер графа
        ver -- массив (N - 1) x M вещественных чисел, параметры парных
            потенциалов вертикальных рёбер графа
        metric -- массив K x K, метрика на K точках.
        Используется при определении парных потенциалов
        """
        self.N, self.M, self.K = unr.shape
        self.cut, self.energy_const = 0, 0
        self.unr, self.hor, self.ver = unr.astype(float), hor, ver
        self.metric = metric


    def energy(self, labels):
        """
        Для данного набора значений переменных графической модели вычисляет энергию.

        Вход:
        labels -- массив N x M целых чисел от 0 до K - 1, значения переменных графической модели
        Выход:
        energy -- вещественное число, энергия на данном наборе переменных
        """
        mask = np.eye(self.K, dtype=bool)[labels]
        s_unr = np.sum(self.unr[mask])
        s_ver = np.sum(self.metric[labels[:-1, :], labels[1:, :]]*self.ver)
        s_hor = np.sum(self.metric[labels[:, :-1], labels[:, 1:]]*self.hor)
        return s_unr + s_ver + s_hor
        

    def expand_alpha(self, labels, alpha):
        """
        Вычисляет один шаг алгоритма alpha_expansion.
        Для данного набора значений переменных labels минимизирует энергию,
        расширяя множество принимающих значение alpha переменных.

        Вход:
        labels -- массив N x M целых чисел от 0 до K - 1, значения переменных графической модели
        alpha -- целое число от 0 до K - 1, параметр расширения
        Выход:
        new_labels -- массив N x M целых чисел от 0 до K - 1, новый набор значений переменных
        """
        terminal_weights, edge_weights = self._build_graph(labels, alpha)
        cut, alpha_labels = graph_cut.graph_cut(terminal_weights, edge_weights)
        alpha_labels.resize(self.N, self.M)
        new_labels = labels.copy()
        new_labels[alpha_labels == 1] = alpha
        self.cut = cut + np.sum(self.energy_const)
        return new_labels
        

    def _build_graph(self, labels, alpha):
        """
        Вспомогательная процедура для метода expand_alpha.
        Для данного набора значений переменных labels и значения alpha сводит
        задачу минимизации энергии к поиску минимального разреза в графе. 

        Вход:
        labels -- массив N x M целых чисел от 0 до K - 1, значения переменных графической модели
        alpha -- целое число от 0 до K - 1, параметр расширения
        Выход:
        terminal_weights -- массив N * M x 2 вещественных чисел,
            пропускные способности терминальных рёбер графа
        edge_weights -- массив (N - 1) * M + N * (M - 1) x 4 вещественных чисел,
            пропускные способности рёбер, соединяющих нетерминальные вершины
        """
                
        N, M, K = self.unr.shape
        
        mask = ((np.arange(N * M - 1) + 1) % M).astype(bool)
        hor_from_to = (np.repeat(np.arange(N * M), repeats=2)[1:-1].
                        reshape((N * M - 1), 2)[mask, :])
        
        mask = np.repeat(np.repeat(np.arange(M)[:, None], 
                        repeats=(N - 1), axis=0), repeats=2, axis=1)
        ver_from_to = (np.tile(np.repeat(np.arange(M * N, step=M), repeats=2)[1:-1].
                        reshape(N - 1, 2), reps=(M, 1)) + mask)
                                
        hor_t00 = self.metric[labels[:, :-1], labels[:, 1:]] * self.hor
        ver_t00 = self.metric[labels[:-1, :], labels[1:, :]] * self.ver
        hor_t01 = self.metric[labels[:, :-1], alpha] * self.hor
        ver_t01 = self.metric[labels[:-1, :], alpha] * self.ver
        hor_t10 = self.metric[alpha, labels[:, 1:]] * self.hor
        ver_t10 = self.metric[alpha, labels[1:, :]] * self.ver
                     
        unr_t1 = self.unr.copy()[:, :, alpha]
        mask = np.eye(K, dtype=bool)[labels, :]
        unr_t0 = self.unr.copy()[mask].reshape(N, M)

        unr_t0[:, :-1] += hor_t00
        unr_t0[:-1, :] += ver_t00
        unr_t1[:, 1:] += hor_t01 - hor_t00
        unr_t1[1:, :] += ver_t01 - ver_t00
        unr_t1[:, :-1] += hor_t00 - hor_t01
        unr_t1[:-1, :] += ver_t00 - ver_t01
               
        hor_t10 += hor_t01 - hor_t00
        ver_t10 += ver_t01 - ver_t00
        
        self.energy_const = np.minimum(unr_t1, unr_t0)
        unr_t1 -= self.energy_const
        unr_t0 -= self.energy_const
        
        terminal_weights = np.stack((unr_t1.ravel(order="C"),
                                     unr_t0.ravel(order="C")), axis=1)
                            
        edge_weights = np.vstack((
            np.hstack((hor_from_to, np.zeros(N * (M - 1))[:, None],
                       hor_t10.ravel(order="C")[:, None])),
            np.hstack((ver_from_to, np.zeros((N - 1) * M)[:, None],
                       ver_t10.ravel(order="F")[:, None]))))

        return terminal_weights, edge_weights


def alpha_expansion(gm, labels, max_iter=50, display=False, rand_order=True):
    """
    Алгоритм alpha_expansion.

    Вход:
    gm -- объект класса GM
    labels -- массив N x M целых чисел от 0 до K - 1, начальные значения переменных
    max_iter -- целое число, максимальное число итераций алгоритма
    display -- булева переменная, алгоритм выводит вспомогательную информацию при display=True
    rand_order -- булева переменная,
        при rand_order=True на каждой итерации alpha берется в случайном порядке
    Выход:
    new_labels -- массив N x M целых чисел от 0 до K - 1, значения переменных,
        на которых завершил работу алгоритм
    energies -- массив max_iter вещественных чисел,
        значения энергии перед каждой итерацией алгоритма
    times -- массив max_iter вещественных чисел, время вычисления каждой итерации алгоритма
    """
    energies, times = np.empty(max_iter), np.empty(max_iter)
    if display:
        print(gm.cut, gm.energy(labels))
    for j in range(max_iter):
        t = time.clock()
        alpha = npr.randint(gm.K) if rand_order else j % gm.K
        labels = gm.expand_alpha(labels, alpha)
        energies[j] = gm.cut
        if display:
            print(alpha)
            print(gm.cut, gm.energy(labels))
            print(labels)
        times[j] = time.clock() - t
    return labels, energies, times
    
    
def define_energy(images, seeds):
    """
    Вычисляет параметры графической модели для K изображений images 
    разрешения N x M и набора семян seeds.    
    
    Вход:
    images -- массив N x M x C x K вещественных чисел, C - число каналов.
    seeds -- массив N x M x K булевых переменных, seeds[n, m, k] = True должно
        поощрять выбор k изображения на позиции n x m при склеивании изображений.
    Выход:
    unr -- массив N x M x K вещественных чисел, унарные потенциалы
    hor -- массив N x (M - 1) вещественных чисел,
        параметры парных потенциалов горизонтальных рёбер графа
    ver -- массив (N - 1) x M вещественных чисел,
        параметры парных потенциалов вертикальных рёбер графа
    metric -- массив K x K, метрика на K точках.
        Используется при определении парных потенциалов
    """
    N, M, C, K = images.shape
    hor = np.diff(images, axis=1).std(axis=3).mean(axis=2)
    ver = np.diff(images, axis=0).std(axis=3).mean(axis=2)
    metric = np.logical_not(np.eye(K, dtype=bool)).astype(int)
    return seeds, hor, ver, metric


def stitch_images(images, seeds):
    """
    Процедура для склеивания изображений.

    Вход:
    images -- массив N x M x C x K вещественных чисел, C - число каналов.
    seeds -- массив N x M x K булевых переменных, seeds[n, m, k] = True должно
        поощрять выбор k изображения на позиции n x m при склеивании изображений.
    Выход:
    res -- массив N x M x C вещественных чисел, склеенное изображение
    mask -- массив N x M целых чисел от 0 до K - 1, mask[n, m] = k равен номеру изображения,
        из которого взят пиксель на позиции n x m.
    """
    N, M, C, K = images.shape
    unr, hor, ver, metric = define_energy(images, seeds)
    labels, energies, times = alpha_expansion(
        GM(unr, hor, ver, metric), npr.choice(K, (N, M)),
        display=True, rand_order=False, max_iter=10)
    mask = (np.eye(K, dtype=bool)[labels][:, :, None, :] * 
            np.ones((N, M, C, K), dtype=bool))
    return images[mask].reshape(N, M, C), labels
    
    
def model_images(path=("my1.png", "my2.png", "mys1.png", "mys2.png")):
    """
    Процедура построения коллажа из двух изображений.
    
    Вход:
    path -- пути файловой системы к изображениям и их семенам
    Выход:
    image -- построенный коллаж
    labels -- финальные метки изображений
    """
    m1 = plt.imread(path[0])
    m2 = plt.imread(path[1])
    images = np.stack((m1, m2), axis=3)
    
    s1 = np.mean(plt.imread(path[2]), axis=2) == 0
    s2 = np.mean(plt.imread(path[3]), axis=2) == 0
    seeds = np.logical_not(np.stack((s1, s2), axis=2))
                            
    image, labels = stitch_images(images, seeds)
    
    plt.imshow(image)    
    return image, labels
    
    
def model_base():
    """
    Отладочная процедура минимизации энергии.
    Требуется монотонное невозрастание энергии,
    и совпадение энергий, рассчитанных разными процедурами.
    """
    test_mask = np.ones((10, 10), dtype=int)
    test_mask[:5, :5] = 0
    test_mask[5:, :5] = 1
    test_mask[5:, 5:] = 2
    test_mask[:5, 5:] = 3
    unary = np.ones((10, 10, 4), dtype=bool)
    unary[:5, :5, 0] = False
    unary[5:, :5, 1] = False
    unary[5:, 5:, 2] = False
    unary[:5, 5:, 3] = False
    metric = np.invert(np.eye(4, dtype=bool))
    vertC = np.zeros((9, 10))
    horC = np.zeros((10, 9))
    gm = GM(unary, horC, vertC, metric)
    labels = np.random.choice(4, size=(10, 10))
    mask, energies, times = alpha_expansion(gm, labels,
                                            rand_order=False, display=True)
                                            
    print(np.all(np.isclose(mask, test_mask)))

#model_images()
#model_base()
