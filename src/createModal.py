import numpy as np  # helps with the math
import pandas as pd

# Randomly permute [0,N] and extract indices for each fold
from src.NeuralNetwork import Network


def crossval_folds(N, n_folds, seed=1):
    np.random.seed(seed)
    idx_all_permute = np.random.permutation(N)
    N_fold = int(N / n_folds)
    idx_folds = []
    for i in range(n_folds):
        start = i * N_fold
        end = min([(i + 1) * N_fold, N])
        idx_folds.append(idx_all_permute[start:end])
    return idx_folds


def parsePeople():
    normalize = True
    target_name = "depressed"
    df = pd.read_csv('data/foreveralone.csv', delimiter=",", dtype={target_name: str})
    target2idx = {target: idx for idx, target in enumerate(sorted(list(set(df[target_name].values))))}
    X = df.drop([target_name], axis=1).values
    personVals = []
    for col in df:
        if col == 'job_title' or col == 'depressed' or col == 'time' or col == 'social_fear' or col == 'what_help_from_others' or col == 'attempt_suicide' or col == 'improve_yourself_how':
            continue
        enum = {target: idx for idx, target in enumerate(sorted(list(set(df[col].values))))}
        personVals.append(np.vectorize(lambda x: enum[x])(df[col].values))

    personVals = np.array(personVals).T
    # X = df.drop([target_name], axis=1).values
    y = np.vectorize(lambda x: target2idx[x])(df[target_name].values)
    n_classes = len(target2idx.keys())
    if X.shape[0] != y.shape[0]:
        raise Exception("X.shape = {} and y.shape = {} are inconsistent!".format(X.shape, y.shape))
    if normalize:
        personVals = (personVals - personVals.mean(axis=0)) / personVals.std(axis=0)
    return personVals, y, n_classes, df


def parsePerson(person_str, df):
    person_str = '9/4/2016 23:10:04,Male,Straight,28,"$40,000 to $49,999",White non-Hispanic,Normal weight,Yes,No,' \
                 'Yes but I ' \
                 'haven\'t, 3.0, Yes, Yes, "wingman/wingwoman, Set me up with a date, date coaching", No, Employed for ' \
                 'wages, Scientist, Master’s degree, Therapy '
    person_str = 'time,gender,sexuallity,age,income,race,bodyweight,virgin,prostitution_legal,pay_for_sex,friends,' \
                 'social_fear,depressed,what_help_from_others,attempt_suicide,employment,job_title,edu_level,' \
                 'improve_yourself_how\n' + person_str
    # file = open('temp.csv', 'w+')
    # file.write(unicode(str, errors='replace'))
    # file.close()
    target_name = "depressed"
    temp_person = pd.read_csv('temp.csv', delimiter=",", dtype={target_name: str})

    personVals = []
    for col in temp_person:
        if col == 'job_title' or col == 'depressed' or col == 'time' or col == 'social_fear' or col == 'what_help_from_others' or col == 'attempt_suicide' or col == 'improve_yourself_how':
            continue
        enum = {target: idx for idx, target in enumerate(sorted(list(set(df[col].values))))}
        personVals.append(np.vectorize(lambda x: enum[x])(temp_person[col].values))

    personVals = np.array(personVals).T
    # X = df.drop([target_name], axis=1).values

    # personVals = (personVals - personVals.mean(axis=0)) / personVals.std(axis=0)
    return personVals


if __name__ == '__main__':
    hidden_layers = [5]  # number of nodes in hidden layers i.e. [layer1, layer2, ...]
    eta = 0.1  # learning rate
    n_epochs = 400  # number of training epochs
    n_folds = 4  # number of folds for cross-validation
    seed_crossval = 1  # seed for cross-validation
    seed_weights = 1  # seed for NN weight initialization

    # Read csv data + normalize features
    print("Reading '{}'...".format('data/foreveralone.csv'))
    X, y, n_classes, df = parsePeople()
    print(" -> X.shape = {}, y.shape = {}, Number of classes = {}\n".format(X.shape, y.shape, n_classes))
    N, d = X.shape

    print("Neural network model:")
    print(" input dimensions:= {}".format(d))
    print(" hidden_layers = {}".format(hidden_layers))
    print(" output dimensions = {}".format(n_classes))
    print(" eta = {}".format(eta))
    print(" n_epochs = {}".format(n_epochs))
    print(" n_folds = {}".format(n_folds))
    print(" seed_crossval = {}".format(seed_crossval))
    print(" seed_weights = {}\n".format(seed_weights))

    # Create cross-validation folds
    idx_all = np.arange(0, N)
    idx_folds = crossval_folds(N, n_folds, seed=seed_crossval)  # list of list of fold indices

    # Train/evaluate the model on each fold
    acc_train, acc_valid = list(), list()

    print()
    print("Cross-validating with {} folds...".format(len(idx_folds)))
    for i, idx_valid in enumerate(idx_folds):
        # Collect training and test data from folds
        idx_train = np.delete(idx_all, idx_valid)
        X_train, y_train = X[idx_train], y[idx_train]
        X_valid, y_valid = X[idx_valid], y[idx_valid]

        # Build neural network classifier model and train
        model = Network(input_dim=d, output_dim=n_classes,
                        hidden_layers=hidden_layers, seed=seed_weights)
        model.train(X_train, y_train, eta=eta, n_epochs=n_epochs)
        print(model.predict(parsePerson('sa', df)))
        # Make predictions for training and test data
        ypred_train = model.predict(X_train)
        ypred_valid = model.predict(X_valid)

        # Compute training/test accuracy score from predicted values
        acc_train.append(100 * np.sum(y_train == ypred_train) / len(y_train))
        acc_valid.append(100 * np.sum(y_valid == ypred_valid) / len(y_valid))

        # Print cross-validation result
        print(" Fold {}/{}: acc_train = {:.2f}%, acc_valid = {:.2f}% (n_train = {}, n_valid = {})".format(
            i + 1, n_folds, acc_train[-1], acc_valid[-1], len(X_train), len(X_valid)))

    # Print results
    print("  -> acc_train_avg = {:.2f}%, acc_valid_avg = {:.2f}%".format(
        sum(acc_train) / float(len(acc_train)), sum(acc_valid) / float(len(acc_valid))))
