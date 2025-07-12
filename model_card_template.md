# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- **Model type**: `AdaBoostClassifier` (ensemble of decision‐tree stumps)  
- **Framework**: scikit-learn (`sklearn.ensemble.AdaBoostClassifier`)  
- **Hyperparameters**:  
  - `n_estimators`: 120  
  - `learning_rate`: 0.5  
  - `random_state`: 26  

## Intended Use
This model predicts whether an individual’s annual income is **">50K"** or **"<=50K"** based on U.S. Census features.  
- **Primary use case**: exploratory analysis or prototyping of income classification models.  
- **Not for** making high-stakes decisions (e.g., loan approvals, hiring) without thorough bias and fairness auditing.

## Training Data
- **Source**: Publicly available U.S. Census Bureau “Adult” dataset (commonly called the “Census Income” dataset).  
- **Features used**:  
  - Continuous: age, hours-per-week, capital-gain, capital-loss, education-num, etc.  
  - Categorical: workclass, education, marital-status, occupation, relationship, race, sex, native-country  
- **Preprocessing**:  
  - Missing categorical values imputed with most frequent category.  
  - One-hot encoding applied to all categorical features.  
  - Continuous features passed through unchanged.  
  - Label (“salary”) binarized to 1 if `>50K`, else 0.

## Evaluation Data
- **Split method**: stratified 80/20 train/test split on the original dataset.  
- **Test set size**: 20% of the full dataset, held out during training.  
- **Slice evaluation**: performance measured on subgroups of each categorical feature (e.g., by `education`, `sex`, `race`).

## Metrics
_Please include the metrics used and your model's performance on those metrics._
- **Precision**: 0.XXXX  
- **Recall**: 0.XXXX  
- **F1-score**: 0.XXXX  

(_Replace “0.XXXX” with the actual values from your test run._)

## Ethical Considerations
- **Potential biases**:  
  - The dataset is known to reflect historical socioeconomic disparities.  
  - Model may exhibit lower performance on underrepresented groups (e.g., certain races, countries).  
- **Fairness auditing**:  
  - We compute precision, recall, and F1 on each subgroup slice.  
  - Use these slice metrics to detect and mitigate unfair outcomes.

## Caveats and Recommendations
- **Caveats**:  
  - This model is trained on census data collected in the 1990s and may not generalize to current populations.  
  - Performance on minority subgroups may be unstable if slice counts are small.  
- **Recommendations**:  
  - Retrain or recalibrate on more recent, diverse data before deployment.  
  - Incorporate additional fairness constraints or post-processing steps if used in decision-making.  
  - Monitor slice performance in production and retrain periodically.  
