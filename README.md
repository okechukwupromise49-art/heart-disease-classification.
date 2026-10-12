# Heart Disease Classification Using Machine Learning

## Project Overview
This project uses machine learning classification algorithms to identify the presence or absence of heart disease based on 13 clinical features from the UCI Heart Disease dataset.

The goal is to compare different machine learning models and evaluate their performance using standard classification metrics.

## Dataset
- **Source:** [UCI Machine Learning Repository — Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Features:** 13 clinical attributes
- **Target:** Presence or absence of heart disease

## Machine Learning Models
The following models were implemented and evaluated:

1. Logistic Regression
2. Random Forest Classifier
3. Multi-Layer Perceptron (MLP) Neural Network

## Evaluation Metrics
The models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Model Performance

| Metric | Logistic Regression | Random Forest | MLP Neural Network |
|---|---:|---:|---:|
| Accuracy | 83.33% | 85.00% | 81.67% |
| Precision | 84.62% | 88.00% | 94.74% |
| Recall | 78.57% | 78.57% | 64.29% |
| F1-score | 81.48% | 83.02% | 76.60% |

Based on these results, Random Forest achieved the highest accuracy and F1-score among the three models. However, model selection should also consider recall and the consequences of false-negative predictions.

## Technologies Used
- Python
- Google Colab
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Project Workflow
1. Load the dataset.
2. Explore and visualize the data.
3. Handle missing values and prepare the target variable.
4. Split the data into training and testing sets.
5. Train the classification models.
6. Evaluate and compare model performance.
7. Interpret the results.

## How to Run
1. Clone or download this repository.
2. Open the `.ipynb` notebook in Google Colab or Jupyter Notebook.
3. Install any required dependencies if necessary.
4. Run the notebook cells in order.

## Limitations
This project is an educational machine learning classification study. Its predictions are based on the dataset used for training and testing. It is not a clinically validated diagnostic tool and should not replace professional medical evaluation.

## Future Improvements
- Investigate F2-score to give greater weight to recall.
- Tune model hyperparameters.
- Apply cross-validation for more robust evaluation.
- Explore model interpretability.
- Develop a simple web application for demonstration.

## Author
**Okechukwu Promise**

Machine Learning and AI Learner | Physics Student | Software Developer

---

*This project was developed for educational purposes to explore machine learning classification and model evaluation.*
