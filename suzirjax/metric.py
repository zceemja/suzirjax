from abc import ABCMeta, abstractmethod

from jax.numpy import ndarray, log2, arange, nonzero, stack, sqrt, sum, take, mean, max, log, exp, log1p

from suzirjax.gui import Connector
from suzirjax.modulation import Modulation

METRIC_METHODS = []


def register_metric(cls):
    METRIC_METHODS.append(cls)
    return cls


class MetricMethod(metaclass=ABCMeta):
    NAME = ""
    METHOD = ""
    UNIT = ""

    def __init__(self, data):
        self.data: Connector = data

    @classmethod
    @property
    def label(cls):
        return f'{cls.NAME} {cls.METHOD}'

    @abstractmethod
    def __call__(self, const: ndarray, nx: ndarray, tx_seq: ndarray, snr: float) -> ndarray:
        raise NotImplemented


def bitmap_indices(bmap: ndarray) -> ndarray:
    a, b = nonzero(bmap.T)
    return stack([b[a == i] for i in arange(bmap.shape[1])]).T


@register_metric
class GMIMaxLog(MetricMethod):
    NAME = "GMI"
    METHOD = "2D Max-Log"
    UNIT = "bit/2D Symbol"

    def __init__(self, data):
        super().__init__(data)
        self.bmap_idx_false = None
        self.bmap_idx_true = None
        self.bmap = None
        self.M = None
        data.on('mod_points', self.update_points)

    def update_points(self, M):
        self.M = M
        m = int(log2(self.M))
        self.bmap = (arange(self.M)[:, None] >> arange(m) & 1).astype(bool)
        self.bmap_idx_true = bitmap_indices(self.bmap)
        self.bmap_idx_false = bitmap_indices(~self.bmap)

    def log_likelihood_ratios(self, sll):
        max_true = max(sll[:, self.bmap_idx_true], axis=1)
        max_false = max(sll[:, self.bmap_idx_false], axis=1)
        return max_true - max_false

    def __call__(self, const: Modulation, nx: ndarray, tx_seq: ndarray, snr: ndarray) -> ndarray:
        """
        Computes the generalized mutual information (GMI) using the max-log approximation.
        """
        sigma = (10 ** (-snr / 20)).mean()
        scaling = sqrt(mean(sum(abs(const.complex) ** 2, axis=1) / (const.modes / 2)))  # / (self.D / 2)
        # const /= scaling
        nx /= scaling

        tx_bits = take(self.bmap, tx_seq[0], axis=0)
        tx = take(const.regular[:, :2], tx_seq[0], axis=0)
        rx = tx + nx

        # Compute the squared distance between the received and constellation points
        squared_distance = ((rx[:, None, :] - const.regular[None, :, :2]) ** 2).sum(axis=-1)

        # Compute the symbol likelihood based on the squared distance and the variance sigma
        symbol_log_likelihood = -squared_distance / sigma ** 2

        # Compute the log likelihood ratios for each sample
        log_likelihood_ratios = self.log_likelihood_ratios(symbol_log_likelihood)

        # Compute the information loss for each sample
        information_loss = mean(log1p(exp((1 - 2 * tx_bits) * log_likelihood_ratios)), axis=0)

        # the GMI is the
        return sum(1 - information_loss / log(2))


@register_metric
class GMILogSum(GMIMaxLog):
    METHOD = "2D Log-Sum"

    def log_likelihood_ratios(self, sll):
        log_sum_true = log(sum(exp(sll[:, self.bmap_idx_true]), axis=1))
        log_sum_false = log(sum(exp(sll[:, self.bmap_idx_false]), axis=1))
        return log_sum_true - log_sum_false


@register_metric
class MutualInformation(MetricMethod):
    NAME = "MI"
    METHOD = "2D"
    UNIT = "bit/2D Symbol"

    def __call__(self, const: Modulation, nx: ndarray, tx_seq: ndarray, snr: float) -> ndarray:
        # scaling = sqrt(mean(sum(const ** 2, axis=1)))
        # const /= scaling
        # nx /= scaling
        sigma = 10 ** (-snr / 20)

        # Dims: M, L, M
        dist = const.regular[None, None, :, :] - const.regular[:, None, None, :]
        c1 = (abs(dist) ** 2).sum(-1)
        c2 = dist * nx[None, :, None]
        info_loss = exp(-(c1 + 2 * c2[:, :, :, 0]) / sigma).sum(axis=-1)

        # Mutual information
        return log2(const.points) - log2(info_loss).mean()


@register_metric
class MutualInformation(MetricMethod):
    NAME = "MI"
    METHOD = "2D Fast"
    UNIT = "bit/2D Symbol"

    def __call__(self, const: ndarray, nx: ndarray, tx_seq: ndarray, snr: float) -> ndarray:
        # scaling = sqrt(mean(sum(const ** 2, axis=1)))
        # const /= scaling
        # nx /= scaling
        sigma = 10 ** (-snr / 20)
        tx = take(const, tx_seq, axis=0)
        rx = tx + nx

        squared_distance = ((rx[:, None, :] - const[None, :, :]) ** 2).sum(axis=-1)
        noise_power = (abs(nx) ** 2).sum(axis=1)[:, None]
        info_loss = exp(-(squared_distance - noise_power) / sigma).sum(axis=1)

        # Mutual information
        return log2(const.shape[0]) - log2(info_loss).mean()

        ### Fast
        # tmp += np.exp(-(abs(sig[l]-symbols[j])**2 - abs(sig[l]-sig_tx[l])**2)/N0)

        ### Not Fast
        # M = symbols.size
        # L = noise.size
        # for i in range(M):
        #     for l in range(L):
        #         tmp = 0
        #         for j in range(M):
        #             tmp += np.exp(
        #                 -(
        #                 abs(symbols[i] - symbols[j]) ** 2
        #                 + 2 * np.real((symbols[i] - symbols[j]) * noise[l])
        #                 ) / N0
        #                 )
        #         mi_out += np.log2(tmp)
        # return np.log2(M) - mi_out / M / L
