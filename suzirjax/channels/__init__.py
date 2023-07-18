from .channel import Channel
from .awgn import AWGNChannel, PCAWGNChannel
from .chalmers import ChalmersQAMpy
from .remote import RemoteChannel
from .ssfm import NonLinearChannel
CHANNELS = [AWGNChannel, PCAWGNChannel, ChalmersQAMpy, NonLinearChannel, RemoteChannel]

