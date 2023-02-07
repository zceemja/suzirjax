#!/usr/bin/env python3
#
# This is a simple Vivado proxy server. When in server mode, it opens vivado in tcl mode and
# forwards commands forwards and backwards from tcp socket.
# Socket protocol is simple: header is 5 bytes followed by payload.
# First 4 header bytes is payload length, 1 after is Vivado exit status (in case it exits or errors)
# exit status code is only valid if payload length is equal to 0xFFFFFFFF
#

import pty
import socket
import sys
import shutil
import os
import atexit
import time
from subprocess import Popen
import select
import psutil
import argparse


def get_address(name):
    host, port = name.split(':', 1)
    return host, int(port)


class FileReader:
    def __init__(self, fileno):
        self._fileno = fileno

    def fileno(self):
        return self._fileno


class VivadoServer:
    def __init__(self, address, vivado_bin, lockfile, logfile, debug=False):
        self.address = address
        self.lockfile = lockfile
        self.logfile = logfile
        self.vivado_bin = vivado_bin
        self.debug_flag = debug
        self.slave = None
        self.pidfile = False
        self.master = None
        self.proc = None

    def close_vivado(self):
        print("=" * 80)
        print('{:^80}'.format(" === Closing vivado === "))
        os.write(self.master, b'\r\nexit\r\n')
        list(self._read_msg(self.master))  # wait until close
        self.proc.terminate()
        self.proc.wait()
        os.close(self.master)
        os.close(self.slave)
        if self.pidfile:
            os.remove(self.lockfile)
        print("=" * 80)

    def debug(self, *s):
        if self.debug_flag:
            print(*s)

    def server_loop(self, output=True):

        if self.pidfile:
            with open(self.lockfile, 'w') as pid_file:
                pid_file.write(str(os.getpid()))

        atexit.register(self.close_vivado)

        with open(self.logfile, 'a') as logf:

            def log(*s):
                logf.write(' '.join(map(str, s)) + '\n')

            if output:
                log = print
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(get_address(self.address))
                sock.listen()
                print('{:^80}'.format(f" == Server started on {self.address} == "))
                print("=" * 80)

                procu = psutil.Process(self.proc.pid)
                status = procu.status()

                while status == psutil.STATUS_SLEEPING or status == psutil.STATUS_RUNNING:
                    r, w, e = select.select([sock], [], [], 0.5)
                    if sock not in r:
                        status = procu.status()
                        continue
                    conn, addr = sock.accept()
                    self.debug("Connection from", ':'.join(map(str, addr)))
                    try:
                        with conn:
                            payload_len = int.from_bytes(conn.recv(4), 'big')
                            payload = conn.recv(payload_len)
                            print("=> ", payload.decode().strip())
                            if not payload.endswith(b'\r\n'):
                                payload += b'\r\n'
                            os.write(self.master, payload)
                            lines = self._read_msg(self.master)
                            for line in lines:
                                # Ignore echo
                                if line.strip() == b'Vivado% ' + payload.strip() or line.strip() == payload.strip():
                                    continue
                                print(line.decode().strip())
                                conn.send(len(line).to_bytes(4, byteorder='big') + b'\x00' + line)
                            print("=" * 80)
                            conn.send(b'\xFF' * 4 + b'\x00')
                    except ConnectionRefusedError:
                        log("Client Connection refused")
                    except KeyboardInterrupt:
                        log("Interrupt received!")
                    status = procu.status()
                log("Vivado ended with status", status)

    def _read_msg(self, fileno):
        buf = b''
        f = FileReader(fileno)
        procu = psutil.Process(self.proc.pid)
        status = procu.status()
        while status == psutil.STATUS_SLEEPING or status == psutil.STATUS_RUNNING:
            r, w, e = select.select([f], [], [], 0.5)
            if f not in r:
                status = procu.status()
                continue
            buf += os.read(fileno, 1024)
            lines = buf.split(b'\r\n')
            if len(lines) > 1:
                for line in lines[:-1]:
                    if line == b'Vivado% ':
                        continue
                    yield line.strip()
            if lines[-1] == b'Vivado% ':
                return
            buf = lines[-1]
            if len(buf) > 0:  # Finish reading
                continue
            status = procu.status()

    def start_server(self, command, vdir, daemon):
        env = {
            'HOME': os.environ['HOME'],
            'XILINXD_LICENSE_FILE': os.environ.get('XILINXD_LICENSE_FILE', ''),
        }
        if not os.path.exists(vdir):
            os.makedirs(vdir)

        self.master, self.slave = pty.openpty()
        self.proc = Popen([self.vivado_bin, '-nolog', '-mode', 'tcl'], cwd=vdir, env=env,
                          stdout=self.slave, stdin=self.slave, stderr=self.slave)
        print(f"Starting {self.vivado_bin} [{self.proc.pid}]")

        for line in self._read_msg(self.master):
            print(line.decode())
        if len(command) > 0:
            os.write(self.master, command.encode() + b'\r\n')
            for line in self._read_msg(self.master):
                print(line.decode())
        print("=" * 80)

        if daemon:
            try:
                from daemonize import Daemonize
                self.pidfile = False
                daemon = Daemonize(
                    app="vivado_proxy", pid=self.lockfile,
                    action=lambda: self.server_loop(output=False)
                )
                daemon.start()
            except ImportError:
                print("Module 'daemonize' is not installed, running in foreground")
                self.pidfile = True
                self.server_loop(output=False)
        else:
            self.pidfile = True
            self.server_loop(output=True)


def send_command(bind: str, command: str):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect(get_address(bind))
    except ConnectionRefusedError:
        print("Connection refused")
        exit(1)
    # print(">", command)
    command = command.encode()
    sock.send(len(command).to_bytes(4, byteorder='big') + command)
    error = False
    while True:
        header = sock.recv(5)
        if len(header) != 5:
            break
        if header[:4] == b'\xFF\xFF\xFF\xFF':
            if header[4] == 0 and error:
                exit(1)
            exit(header[4])
        payload_len = int.from_bytes(header[:4], 'big')
        payload = sock.recv(payload_len)
        if payload is None:
            break
        for line in payload.split(b'\r\n'):
            if line.strip().startswith(b'ERROR:'):
                error = True
            print(line.decode())
    return error

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='vivado_proxy',
        description='Proxy for Vivado TCL server',
    )
    parser.add_argument('-a', '--address', help="Bind address", default='127.0.0.1:4999')
    parser.add_argument('-b', '--vivado', help="Vivado executable", default='vivado')
    parser.add_argument('-s', '--server', action='store_true', help="Start a server")
    parser.add_argument('-D', '--daemon', action='store_true', help="Run server as a daemon")
    parser.add_argument('-v', '--debug', action='store_true', help="Print verbose debugging")
    parser.add_argument('-l', '--lock', default='vivado_proxy.lck', help="Server lock file")
    parser.add_argument('-L', '--log', default='vivado_proxy.log', help="vivado proxy log file for server")
    parser.add_argument('-d', '--vdir', default='.', help="Vivado journal/log directory")
    parser.add_argument('-r', '--retry', default=5, type=int,
                        help="Times to retry until client gives up to connect to server")
    parser.add_argument('command', nargs='*', help='TCL command, if not specified, start server')
    args = parser.parse_args()

    if args.server or len(args.command) == 0:
        if os.path.exists(args.lock):
            pidok = False
            with open(args.lock, 'r') as f:
                try:
                    pid = int(f.read())
                    psutil.Process(pid)
                    print(f"Server is already running [process {pid}]")
                except ValueError:
                    print(f"Server is already running [lockfile {args.lock}]")
                except psutil.NoSuchProcess:
                    pidok = True
            if not pidok:
                exit(1)
        vivado = shutil.which(args.vivado)
        if vivado is None:
            print("Missing vivado is system path!")
            exit(1)
        srv = VivadoServer(args.address, vivado, args.lock, args.log, args.debug)
        srv.start_server(' '.join(args.command), args.vdir, args.daemon)
    else:
        for _ in range(args.retry):
            try:
                if not send_command(args.address, ' '.join(args.command)):
                    exit(0)
            except ConnectionRefusedError:
                time.sleep(1)
        print("Server is not running")
        exit(1)
