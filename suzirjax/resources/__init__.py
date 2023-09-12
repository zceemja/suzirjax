from os import path


def get_resource(name) -> str:
    _cdir = path.dirname(path.realpath(__file__))
    return path.join(_cdir, name)

