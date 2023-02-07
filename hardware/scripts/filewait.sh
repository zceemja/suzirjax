#!/bin/bash -e
# Wait for file to exist
# TODO: use entr or ionotify

until [ -f $1 ]; do
     sleep 1
done
exit
