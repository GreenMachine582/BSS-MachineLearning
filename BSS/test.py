
import logging
import os

import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor

import BSS

# Constants
local_dir = os.path.dirname(__file__)


def trainModel(x_train, y_train, random_seed: int = None):
    # TODO: Fix, Change, remove?
    logging.info("Training best model")
    model = GradientBoostingRegressor(random_state=random_seed)
    model.fit(x_train, y_train)
    return model


def plotPredictions(model, dataset, x_test):
    # TODO: Fix, Change, remove?
    # plots a line graph of BSS True and Predicted Demand vs Date
    predicted_demand = model.model.predict(x_test)
    plt.figure()
    plt.plot(dataset.df.index, dataset.df['cnt'], color='blue')
    plt.plot(x_test.index, predicted_demand, color='red')
    plt.title('BSS Demand Vs Datetime')
    plt.xlabel('Datetime')
    plt.ylabel('Cnt')
    plt.show()


def main(dir_=local_dir):
    # TODO: Documentation and error handling
    config = BSS.Config(dir_, 'Bike-Sharing-Dataset-day')

    dataset = BSS.Dataset(config.dataset)
    if not dataset.load():
        return

    dataset = BSS.processDataset(dataset)

    X_train, X_test, y_train, y_test = dataset.split(config.random_seed)

    model = trainModel(X_train, y_train, random_seed=config.random_seed)
    score = model.score(X_test, y_test)

    model = BSS.Model(config.model, model=model)
    model.save()
    model.resultAnalysis(score, X_test, y_test)

    plotPredictions(model, dataset, X_test)

    logging.info(f"Completed")
    return


if __name__ == '__main__':
    main()
