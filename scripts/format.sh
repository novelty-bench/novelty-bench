#!/bin/bash

ruff check --select E,F,UP,B,SIM,I --ignore E501 --fix
ruff format src/*.py