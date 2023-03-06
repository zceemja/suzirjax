from .channel import Channel
from .awgn import AWGNChannel, PCAWGNChannel
from .chalmers import ChalmersQAMpy
from .remote import RemoteChannel
from .remote import RemoteChannel
CHANNELS = [AWGNChannel, PCAWGNChannel, ChalmersQAMpy, RemoteChannel]

