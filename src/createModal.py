import csv

import matplotlib.pyplot as plt  # to plot error during training
import numpy as np  # helps with the math
from src.NeuralNetwork import NeuralNetwork
from src.Person import Person


def parsePeople():
    persons = []
    # d_parser = lambda x: datetime.strptime(x, '%m/%d/%Y %H:%M:%S')
    # dataset = pd.read_csv('data/foreveralone.csv', parse_dates=['time'],
    #                       date_parser=d_parser)
    # parseAPerson(dataset["gender"])
    with open('../data/foreveralone.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                newPerson = Person(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9],
                                   row[10], row[11], row[12], row[13], row[14], row[15], row[16], row[17], row[18])
                persons.append(newPerson)
                line_count += 1
        return persons


def normalizeFriend(friendNumber):
    friendNumber = float(friendNumber)
    if friendNumber > 100:
        return 1
    else:
        return friendNumber / 100


def parseInputForAPerson(person):
    value = []
    if person.gender == "Male":
        value.append(0)
    else:
        value.append(1)



    # if person.sexuallity == "Straight":
    #     value.append(0)
    # else:
    #     value.append(1)
    #
    # if person.virgin == "Yes":
    #     value.append(0)
    # else:
    #     value.append(1)
    #
    # if person.pay_for_sex == "No":
    #     value.append(0)
    # else:
    #     value.append(1)

    # value.append(normalizeFriend(person.friends))
    return value


def parseOutputForAPerson(person):
    value = []
    if person.depressed == "No":
        value.append(0)
    else:
        value.append(1)
    return value


if __name__ == '__main__':
    persons = parsePeople()
    inputs = []
    outputs = []
    training = persons[0:400]
    tests = persons[401:]
    for person in persons:
        inputs.append(parseOutputForAPerson(person))
        outputs.append(parseOutputForAPerson(person))
    # input data
    # inputs = np.array([[0, 1, 0],
    #                    [0, 1, 1],
    #                    [0, 0, 0],
    #                    [1, 0, 0],
    #                    [1, 1, 1],
    #                    [1, 0, 1]])
    # output data
    # outputs = np.array([[0], [0], [0], [1], [1], [1]])
    NN = NeuralNetwork(np.array(inputs), np.array(outputs))
    # train neural network
    NN.train()

    # print the predictions for both examples
    for person in tests:
        if NN.predict(parseInputForAPerson(person))[0] == 0.07442911:
            print(person)
        print(NN.predict(parseOutputForAPerson(person)), ' - Correct: ', parseOutputForAPerson(person))
        if NN.predict(parseInputForAPerson(person)) < 0.5 and parseOutputForAPerson(person) == 1:
            print(NN.predict(parseInputForAPerson(person)), ' - Correct: ', parseOutputForAPerson(person))
        if NN.predict(parseInputForAPerson(person)) > 0.5 and parseOutputForAPerson(person) == 0:
            print(NN.predict(parseInputForAPerson(person)), ' - Correct: ', parseOutputForAPerson(person))

    # plot the error over the entire training duration
    plt.figure(figsize=(15, 5))
    plt.plot(NN.epoch_list, NN.error_history)
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    # plt.show()
