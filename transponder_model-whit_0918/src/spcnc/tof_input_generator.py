import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import random

distance  = 0.1 # 10 cm
c         = 3.0e8 # speed of light

def generate_tof_event():
  v   = random.uniform(0.01, 1.0)
  tof = distance/(v*c)
  #print(f"v = {v}c, tof: {tof}")
  return {"v": v*c, "tof": tof}

def main() -> int:
  """ Prototype network simulation"""
  generate_tof_event()

if __name__ == '__main__':
  sys.exit(main())  # next section explains the use of sys.exit




