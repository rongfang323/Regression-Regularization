import csv
import pandas as pd
import numpy as np


def getpercentage(data):
    dictionary = {}
    for val in data:
        if val in dictionary:
            dictionary[val] += 1
        else:
            dictionary[val] = 1

    for key in dictionary.keys():
        dictionary[key] = float(dictionary[key] / data.count())
    return dictionary


def normalizematrix(input):
    for i in range(len(input[0])):
        c = input[:, i]
        if np.ptp(c) == 0:
            if c[0] == 0:
                input[:, i] = c
            else:
                input[:, i] = c/c[0]
        else:
            input[:, i] = (c - c.min()) / (np.ptp(c))
    return input


def getloss(x, y, w):
    totalLoss = 0
    for i in range(len(x)):
        yi = np.dot(w, x[i])
        totalLoss += (y[i] - yi) ** 2
    return totalLoss


def gradientdescent(x, y, lr, lambdaReg, iterations, batch_size):
    w = np.random.uniform(-0.2, 0.2, len(x[0]))
    batch_count = len(x) // batch_size

    for it in range(iterations+1):
        for batch_i in range(batch_count):

            p_y = np.matmul(x[batch_i * batch_size: (batch_i + 1) * batch_size], w)
            det_w = np.matmul(np.transpose(x[batch_i * batch_size: (batch_i + 1) * batch_size]), p_y - y)
            #   add the regularization item
            w_for_reg = np.array(w)
            w_for_reg[0] = 0
            det_w += lambdaReg * w_for_reg
            norm = np.linalg.norm(det_w)

            #   threshold for end iteration
            if norm <= 0.5:
                print("iteration ends : " + str(it))
                record = np.array([it, sse])
                return w, record

            sse = getloss(x,y,w)

            if np.isnan(sse) or np.isnan(norm):
                print("Iteration %d diverged" % it)
                mylist = []
                record = np.array([it, 0])
                return np.array(mylist), record

            if (it % 100000 == 0) and batch_i == batch_count - 1:
                print("it = %d, SSE = %f" % (it, sse))
                print("norm = %f" % norm)

            w -= lr * det_w
            record = np.array([it, sse])
    return w, record

def buildstats(data, isTraining):
    ignore = ["waterfront", "condition", "grade", "zipcode"]
    df = pd.read_csv(data)
    df[['month', 'day', 'year']] = df.date.str.split("/", expand=True).astype(float)
    cols = df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('year')))
    cols.insert(1, cols.pop(cols.index('day')))
    cols.insert(2, cols.pop(cols.index('month')))
    df = df.reindex(columns=cols)
    df = df.drop('date', 1)
    df = df.drop('id', 1)
    df = df.drop('dummy', 1)
    #print(df.head())

    # the year renovated has range 0 to 2015.
    # hence setting it year build if value is 0
    for i in range(len(df)):
        if df.iloc[i, 15] == 0:
            df.iloc[i, 15] = df.iloc[i, 14]

    print(df.head())
    v = df.values
    v = v.astype(float)
    if isTraining:
        return v[:, :-1], v[:, -1]
    else:
        return v

def getprediction(x, weights):
    y = []
    # compute dot product to get prediction
    for i in range(len(x)):
        y.append(np.dot(weights, x[i]))
    return y


if __name__ == '__main__':
    #learningrates = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]
    learningrate = 0.00001
    lambdaRegs = [0, 0.001, 0.01, 0.1, 1, 10, 100]
    totaliterations = 1000000
    batchsize = 10000
    training_data, training_label = buildstats('PA1_train.csv', True)
    validation_data, validation_label = buildstats('PA1_dev.csv', True)
    test_data = buildstats('PA1_test.csv', False)
    training_data = normalizematrix(training_data)
    lambdaCount = len(lambdaRegs)

    count=0
    for lambdaReg in lambdaRegs:
        print("lambda = " , lambdaReg)
        count += 1
        result = gradientdescent(training_data, training_label, float(learningrate), float(lambdaReg), totaliterations,
                                  batchsize)
        weights[count] = result.w
        record[count] = result.record
        validation_data = normalizematrix(validation_data)
        test_data = normalizematrix(test_data)
        if weights.size != 0:
            print("weight = " + str(weights))
            print("SSE on validation dataset: %f" % (getloss(validation_data, validation_label, weights)))
            test_labels = getprediction(test_data, weights)
            #   output result to csv
            with open(str(learningrate) + '.csv', mode='w') as file:
                writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                for i in range(len(test_labels)):
                    writer.writerow([test_labels[i]])