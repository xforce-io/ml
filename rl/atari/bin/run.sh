#!/bin/bash

python atari/main.py --algo dqn --env BreakoutNoFrameskip-v4 --steps 1000000 --record-video --mode train