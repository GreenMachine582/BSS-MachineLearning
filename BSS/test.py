
import logging
import os

from sklearn.ensemble import GradientBoostingRegressor

import BSS

# Constants
local_dir = os.path.dirname(__file__)


def main(dir_=local_dir):
    config = BSS.Config(dir_, 'Bike-Sharing-Dataset-hour')

    dataset = BSS.Dataset(config.dataset)
    if not dataset.load():
        return

    dataset = BSS.processDataset(dataset)

    X, y = dataset.getIndependent(), dataset.getDependent()
    X_train, X_test, y_train, y_test = dataset.split(config.random_seed)

    estimator = GradientBoostingRegressor(random_state=config.random_seed)
    model = BSS.Model(config.model, estimator=estimator)

    param_grid = {'loss': ['squared_error', 'absolute_error']}

    cv_results = model.gridSearch(param_grid, X, y)
    estimator = cv_results.best_estimator_
    print('The best estimator:', estimator)
    print('The best score: %.2f%s' % (cv_results.best_score_ * 100, '%'))
    print('The best params:', cv_results.best_params_)

    model.fit(X_train, y_train)
    model.save()

    y_pred = model.predict(X_test)
    model.resultAnalysis(y_test, y_pred)
    model.plotPredictions(dataset.df, X_test, y_pred)

    quit()

    logging.info(f"Completed")
    return


if __name__ == '__main__':
    main()
