# Dataset Description
The competition dataset comprises a set of timeseries with 79 features and 9 responders, anonymized but representing real market data. The goal of the competition is to forecast one of these responders, i.e., responder_6, for up to six months in the future.

You must submit to this competition using the provided Python evaluation API, which serves test set data one timestep by timestep. To use the API, follow the example in this notebook. (Note that this API is different from our legacy timeseries API used in past forecasting competitions.)

# Competition Phases and Data Updates
In line with the forecasting task, the competition will proceed in two phases:

A model training phase with a test set of historical data. This test set has about 4.5 million rows.
A forecasting phase with a test set to be collected after submissions close. You should expect this test set to be about the same size as the test set in the first phase.
To help you author robust submissions, during the final weeks of the model training phase we will be extending the public test set to include data closer to the submission deadline. Predictions on this extended set will not be scored.

At the start of the forecasting phase, the unscored public test set will be extended up to the final day of the model training phase and the private set updated roughly every two weeks. Submissions will be rescored at the time of each update.

During the forecasting phase, the evaluation API will serve test data from the beginning of the public set to the end of the private set. You must make predictions at every timestep, but, in this phase, only predictions on the private set are scored. (You may predict 0.0 on the unscored segments, if you like.)

# File and Field Information
train.parquet - The training set, contains historical data and returns. For convenience, the training set has been partitioned into ten parts.
date_id and time_id - Integer values that are ordinally sorted, providing a chronological structure to the data, although the actual time intervals between time_id values may vary.
symbol_id - Identifies a unique financial instrument.
weight - The weighting used for calculating the scoring function.
feature_{00...78} - Anonymized market data.
responder_{0...8} - Anonymized responders clipped between -5 and 5. The responder_6 field is what you are trying to predict.
test.parquet - A mock test set which represents the structure of the unseen test set. This example set demonstrates a single batch served by the evaluation API, that is, data from a single date_id, time_id pair. The test set contains columns including date_id, time_id, symbol_id, weight, is_scored, and feature_{00...78}. You will not be directly using the test set or sample submission in this competition, as the evaluation API will get/set the test set and predictions.
is_scored - Indicates whether this row is included in the evaluation metric calculation.
lags.parquet - Values of responder_{0...8} lagged by one date_id. The evaluation API serves the entirety of the lagged responders for a date_id on that date_id's first time_id. In other words, all of the previous date's responders will be served at the first time step of the succeeding date.
sample_submission.csv - This file illustrates the format of the predictions your model should make.
features.csv - metadata pertaining to the anonymized features
responders.csv - metadata pertaining to the anonymized responders
Each row in the {train/test}.parquet dataset corresponds to a unique combination of a symbol (identified by symbol_id) and a timestamp (represented by date_id and time_id). You will be provided with multiple responders, with responder_6 being the only responder used for scoring. The date_id column is an integer which represents the day of the event, while time_id represents a time ordering. It's important to note that the real time differences between each time_id are not guaranteed to be consistent.

The symbol_id column contains encrypted identifiers. Each symbol_id is not guaranteed to appear in all time_id and date_id combinations. Additionally, new symbol_id values may appear in future test sets.