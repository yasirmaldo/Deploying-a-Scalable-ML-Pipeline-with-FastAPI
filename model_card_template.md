# Model Card


## Model Details
- **Model type**: `AdaBoostClassifier` 
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
- **Source**: U.S. Census Bureau “Adult” dataset (commonly called the “Census Income” dataset).  
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
- **Slice evaluation**: performance measured on subgroups of each categorical feature 

## Metrics
- **Precision**: 0.7131
- **Recall**: 0.6850  
- **F1-score**: 0.6988  


## Ethical Considerations
- **Potential biases**:  
  - The dataset can potentially reflect socioeconomic disparities.  
  - Model may exhibit lower performance on underrepresented groups.  
- **Fairness auditing**:  
  - We compute precision, recall, and F1 on each subgroup slice.  
  - Use these slice metrics to detect and mitigate unfair outcomes.

## Caveats and Recommendations
- **Caveats**:  
  - This model is trained on census data collected in the past and may not generalize to current populations.  
  - Performance on minority subgroups may be unstable if slice counts are small.  
- **Recommendations**:  
  - Retrain or recalibrate on more recent, diverse data before deployment.  
 
