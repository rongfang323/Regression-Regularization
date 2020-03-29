import numpy as np
import pandas as pd
if __name__ == '__main__':
    w = np.random.uniform(-0.2, 0.2, 8)
    print(w)
    x = [[3,23,5], [1,1,1], [0, 7, 8]]
    batch_count = len(x)
    print("batch_count = " + str(batch_count))
    df = pd.read_csv("PA1_train.csv")
    for batch_i in range(batch_count):
        d = x[batch_i * 2: (batch_i + 1) *2]
        print("batch_i" + str(batch_i))
        print( d)
