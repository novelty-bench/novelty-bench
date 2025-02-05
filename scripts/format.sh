#!/bin/bash

ruff check --select E,F,UP,B,SIM,I --ignore E501 --fix  src/*.py
ruff format src/*.py