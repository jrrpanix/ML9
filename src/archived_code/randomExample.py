import numpy as np
import sys

#
# simple test python
#

def main():
    assert len(sys.argv) > 1
    N = int(sys.argv[1])
    assert N < 1e7
    uni = np.random.uniform(-10,10,N)
    print(np.mean(uni), np.std(uni))
    nrm = np.random.normal(0, np.std(uni), N)
    print(np.mean(nrm), np.std(nrm), N)
    

if __name__ == '__main__':
    main()
