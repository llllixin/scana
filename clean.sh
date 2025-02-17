#!/bin/sh

fd ".DS_Store" . -x rm -rf {};
fd "__pycache__" . -x rm -rf {};
