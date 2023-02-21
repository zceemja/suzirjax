from .awgn import AWGNChannel, PCAWGNChannel
from .chalmers import ChalmersQAMpy
from .remote import RemoteChannel
from .remote import RemoteChannel

CHANNELS = [ChalmersQAMpy, AWGNChannel, PCAWGNChannel, RemoteChannel]