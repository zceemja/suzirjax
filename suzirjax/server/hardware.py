from aiohttp import web
import socketio

sio = socketio.AsyncServer()


@sio.event
def connect(sid, environ):
    print("connect ", sid)


@sio.event
def disconnect(sid):
    print('disconnect ', sid)


@sio.event
async def transmit(sid, data):
    print("transmit ", data)


@sio.on('ping')
async def ping(sid, data):
    print('data:' + data)
    sio.emit('pong', data, to=sid)


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
    sio = socketio.AsyncServer()
    app = web.Application()
    app.router.add_get('/', index)
    sio.attach(app)
    try:
        web.run_app(app, host=sys.argv[1], port=int(sys.argv[2]))
    except ValueError:
        print("Port value invalid")
        exit(1)
