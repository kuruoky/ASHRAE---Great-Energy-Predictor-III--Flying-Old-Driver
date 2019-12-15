This is the readme file for the term project of Group Flying Old Driver (Formerly Team Team)

The project consists of four python files, a pdf report and a .yml environment file.

To proper run this project on an alien PC, please first import the .yml file into a new conda environment for dependecy and packages.

Then please download the dataset(train, test, weather_train, weather_test, submission) on Kaggle and place them in the same directory as the python scripts.

Then please run the preprocess.py which will do a fast preprocessing of the data and transform them into .feather format that allows for fast reading.

Then please run the train2.py which begins the training of the LightGBM. The hyperparameter has already been specified as we run them. If you want to recalculate the hyperparameter, please email me at int.zjy@gmail.com and I will provide you with further script(really time consuming, roughly 3 hours for optimization and search)

Then the trian2.py would generate couple of files, in .dict and .csv. Please do not modify them.

The project that we do does not use leakge data, but if you wish to append it to our result please feel free to add it.

After running train2.py, please run test.py which would perform an operation that deregularize our prediction data to actual prediction that can be submitted.

The additional blending.py is for the heuristic search using leakage data, if you wish to run this please download the leak dataset from the Kaggle forum and some of the publicly available submission and then rename them accordingly. 

Thank you very much!

