#!/bin/bash -e

# Loads specified project if its not loaded yet

DIR=$(dirname -- "$( readlink -f -- "$0"; )")

if (($#!=1)); then
    echo Usage: $0 [project name]
    exit 1
fi
PRONAME=$(basename $1)
PRONAME=${PRONAME%.*}

[[ $($DIR/vivado_proxy.py "current_project") == "$PRONAME" ]] || $DIR/vivado_proxy.py "open_project $1"
