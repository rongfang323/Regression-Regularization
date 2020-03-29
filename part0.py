import pandas as pd

def getpercentage(data):
    dictionary = {}
    for val in data:
        if val in dictionary:
            dictionary[val] += 1
        else:
            dictionary[val] = 0
    sum = 0
    for key in dictionary.keys():
        sum += dictionary[key]
    for key in dictionary.keys():
        dictionary[key] = float(dictionary[key] / sum)
    return dictionary

def buildstats(data):
    ignore = ["waterfront", "condition", "grade", "zipcode"]
    df = pd.read_csv(data, usecols=lambda column: column not in ["dummy", "id"])
    df[['month', 'day', 'year']] = df.date.str.split("/", expand=True).astype(float)
    cols = df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('year')))
    cols.insert(1, cols.pop(cols.index('day')))
    cols.insert(2, cols.pop(cols.index('month')))
    df = df.reindex(columns=cols)
    df = df.drop('date', 1)
    print(df.head())

    for col in df:
        if col in ignore:
            print("col")
            print(getpercentage(df[col]))
            print("\n")
        else:
            print(col)
            print("mean: " + str(df[col].mean(axis=0, skipna=True)))
            print("std: " + str(df[col].std(axis=0, skipna=True)))
            print("min: " + str(df[col].min(axis=0, skipna=True)))
            print("max: " + str(df[col].max(axis=0, skipna=True)))
            print("range: " + str(df[col].max(axis=0, skipna=True) - df[col].min(axis=0, skipna=True)))
            print("\n")


if __name__ == '__main__':
    print("Training stats")
    buildstats('PA1_train.csv')
    print("\n")
    print("-----------------------------------------------------------------------------------------------------------------------")
    print("Dev stats")
    buildstats('PA1_dev.csv')
    print("\n")
    print("-----------------------------------------------------------------------------------------------------------------------")
    print("Test stats")
    buildstats('PA1_test.csv')
    print("\n")
    print(
        "-----------------------------------------------------------------------------------------------------------------------")