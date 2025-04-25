import os
from functools import partial
from multiprocessing import Pool

NUM_PROCESS = 2


def preprocess(arg1, arg2, arg3):

    # for i in range(100):
    #     print(i, arg1)
    print(arg1)

def main():

    d = [f for f in os.listdir('.') if f.endswith('.py')]

    print(d, len(d))

    # Use multiprocessing to split into chunks
    # Create partial function for pool
    f = partial(preprocess, arg2=100, arg3=200)
    p = Pool(NUM_PROCESS)
    p.map(f, d)

    return


if __name__ == "__main__":
    main()