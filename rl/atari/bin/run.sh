#!/bin/bash

source .venv-atari/bin/activate

if [ "$1" == "breakout" ]; then
    python -m atari.main --algo dqn --env BreakoutNoFrameskip-v4 --config breakout --steps 1000000 --record-video --mode train "${@:2}"
elif [ "$1" == "mario" ]; then
    python -m atari.main --game-type mario --world-stage 1-1 --algo dqn --mode train --steps 500000 --record-video "${@:2}"
fi