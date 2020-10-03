#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 23:59:05 2020

@author: nitolapio
"""

### CLASIFICACIÓN DE TEXTO BÁSICA
import matplotlib.pyplot as plt
import os
import re
import shutil
import string  # No sé si es necesario pero yo lo importo por sea caso
import tensorflow as tf
 
from tensorflow import keras

print(tf.__version__)