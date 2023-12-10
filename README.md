
# Forest Cover Type Prediction
A Machine Learning Project to classify 7 forest cover types in 4 wilderness areas.


## Project Overview

### Business Problem:

The goal of the Project is to predict seven different Cover Types in four different Wilderness Areas of the Roosevelt National Forest of Northern Colorado with the best accuracy.

### Machine Learning Problem:

It is a Multi-Class Classification Problem.

### Tasks:

- Task 1: Prepare a complete Data Analysis report on the given data.

- Task 2: Create a Predictive Model which helps to predict seven different Cover Types in four different Wilderness Areas of the Forest with the best accuracy.

## Installation & Setup

### Resources Used

- Editor/IDE: Jupyter Notebook.
- Environment/Backend: Conda
- Python Version: 3.9

### Python Packages Used

- General Purpose: warnings, collections
- Data Wrangling: pandas, numpy
- Data Visualization: matplotlib, seaborn
- Machine Learning: scipy, sklearn

## Data
The actual forest cover type for a given 30 x 30 meter cell was determined from US Forest Service (USFS) Region 2 Resource Information System data. Independent variables were then derived from data obtained from the US Geological Survey and USFS. The data is in raw form (not scaled) and contains binary columns of data for qualitative independent variables such as wilderness areas and soil type.

This study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado. These areas represent forests with minimal human-caused disturbances, so that existing forest cover types are more a result of ecological processes rather than forest management practices.

Data: `forest_train.csv`

## Code Structure
```
├── dataset
│   └── forest_train.csv
├── report
│   ├── Forest Cover Prediction.pdf
│   └── Forest Cover Prediction.pptx
├── src
│   ├── data
│   │   └── forest_train.csv
│   └── Forest Cover Prediction.ipynb
├── .gitignore
└── README.md
```

## Results & Evaluation

### Model Comparison Report
- Random Forest achieved the highest accuracy: 85.88% (Although, Overfit!)

- Best Parameters after Hyperparameter Tuning for Random Forest:
```
RandomForestClassifier(n_estimators=30,criterion='gini', n_jobs=-1, 
                        max_depth=25, min_samples_leaf=4, max_features='log2', bootstrap=True,
                        random_state=4)
```
- Models used for Voting Classifier: [KNN, LogisticRegression, DecisionTree, RandomForests]

### Detailed Model Comparison (Criteria: Accuracy)

| Model | Accuracy on Raw Data | Accuracy on Scaled Data | Accuracy after Removing 5 Features (Pearson Correlation) |
| --- | --- | --- | --- |
| LinearSVC | 0.519637 | 0.589673 | 0.337270 |
| DecisionTreeClassifier | 0.770393 | 0.727547 | 0.768470 |
| LogisticRegression | 0.663279 | 0.644328 | 0.673167 |
| GaussianNB | 0.581159 | 0.581434 | 0.585004 |
| **`RandomForestClassifier`** | 0.855809 | 0.825597 | **`0.858830`** |
| GradientBoostingClassifier | 0.793189 | 0.750893 | 0.795111 |
| KNNeighborsClassifier | 0.792365 | 0.724801 | 0.792365 |
| Voting Classifier ‘Soft Voting’ | 0.815984 | 0.779181 | 0.817632 |
| Voting Classifier ‘Hard Voting’ | 0.836034 | 0.797033 | 0.834660 |

### Conclusion

Forest Cover Prediction Dataset was both tricky and easy in many ways because it had All-Numeric data with most of the features One-Hot-Encoded and No-Missing values. But it also had very Skewed features with 4.25% Outliers(IQR) and Least Correlated Data. With these datapoints, Random Forests Classifier achieved 82% Accuracy when trained on most important 49 features based on Pearson-r value based selection.

## Future Work
XGBoost Algorithm may achieve better accuracy with Hyperparameter Tuning.

## License

[MIT](https://choosealicense.com/licenses/mit/)

