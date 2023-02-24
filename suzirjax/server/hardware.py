import time

from aiohttp import web
import socketio
from channels import ChalmersQAMpy
from channels.remote import bytes_to_array, array_to_bytes

sio = socketio.AsyncServer(async_mode='aiohttp')
app = web.Application()
sio.attach(app)


class Simulator(ChalmersQAMpy):
    def __init__(self, *args):
        super().__init__(*args)
        self.data["fb"] = 15
        self.data["linewidth"] = 100
        self.data["snr"] = 20
        self.data["ntaps"] = 17
        self.data["cpe_en"] = True
        self.data["synced"] = False
        self.data["snr_est"] = '- / -'


sim = Simulator(None)


def bytes2human(n, fmt="{value:.1f}{symbol}B"):
    """ src: https://stackoverflow.com/questions/13343700 """
    symbols = ('', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi')
    prefix = {}
    for i, s in enumerate(symbols[1:]):
        prefix[s] = 1 << (i+1)*10
    for symbol in reversed(symbols[1:]):
        if n >= prefix[symbol]:
            value = float(n) / prefix[symbol]
            return fmt.format(symbol=symbol, value=value)
    return fmt.format(symbol=symbols[0], value=n)

@sio.event
def connect(sid, environ):
    print("connect ", sid)


@sio.event
def disconnect(sid):
    print('disconnect ', sid)


@sio.event
async def config(sid, data: dict):
    for k, v in data.items():
        sim.data[k] = v


@sio.event
async def ping(sid, data):
    await sio.emit('pong', data, to=sid)


@sio.event
async def tx(sid, const_data, tx_data):
    t = time.time_ns()
    const, tx = bytes_to_array(const_data), bytes_to_array(tx_data)
    rx, snr = sim.propagate(const, tx)
    await sio.emit('rx', (array_to_bytes(rx), float(snr)), to=sid)
    t = (time.time_ns() - t) / 1e6
    print(f"Transmission took {t:.2f}ms [{bytes2human(len(const_data)+len(tx_data))}]")


@sio.on('*')
async def catch_all(event, data):
    pass


async def index(request):
    """Serve the client-side application."""
    sids = sio.manager.rooms.get('/', {}).get(None, {})
    return web.Response(text=f"Suzirjax\n\nServer side OK\nClients: {len(sids)}", content_type='text/plain')


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} [host] [port]")
        exit(1)

    # Example taken from https://python-socketio.readthedocs.io/en/latest/intro.html
    app.router.add_get('/', index)
    try:
        web.run_app(app, host=sys.argv[1], port=int(sys.argv[2]))
    except ValueError:
        print("Port value invalid")
        exit(1)
