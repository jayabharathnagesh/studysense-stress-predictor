# StudySense: Student Stress Risk Predictor

A machine learning web app that estimates a student's stress risk level (**Low / Medium / High**) from 20 questions about mental health, study life, environment and social support, and explains **why** it gave that result.

**Live app:** [ADD YOUR STREAMLIT LINK HERE]

> **This is a risk indicator, not a medical diagnosis.** It is a learning project built on a public dataset that appears to be synthetic.

![StudySense screenshot](screenshot.png)

## What it does

* Takes 20 inputs (sliders and a Yes/No question) and predicts Low, Medium or High stress risk.
* Shows the model's confidence for each level, and gives different feedback depending on how confident it is (0-49%, 50-79%, 80-100%).
* Lists the **top 3 reasons** behind the result, each with a practical tip.
* Shows the Tele-MANAS helpline (14416) on High results.
* Has light and dark modes.

## Data

[Student Stress Factors dataset (Kaggle)](https://www.kaggle.com/datasets/rxnach/student-stress-factors-a-comprehensive-analysis): 1,100 students, 20 features, 3 stress levels. It is clean and balanced, with no missing values or duplicates, which suggests it is synthetic. Results show patterns in this dataset, not real-world causes.

A second dataset, [Depression Student Dataset (Kaggle)](https://www.kaggle.com/datasets/ikynahidwin/depression-student-dataset), was explored in the notebook for comparison. I dropped its suicidal-thoughts column and did not use it in the app.

## Method

1. Explored the data and checked correlations with stress level.
2. Split 80/20 with stratification (`random_state=42`), so 880 rows to train and 220 to test.
3. Compared Random Forest and Logistic Regression, then tuned the Random Forest with GridSearchCV.
4. Explained the model with SHAP in the notebook.
5. Built the Streamlit app, with a lightweight explanation method for the top 3 reasons.

## Results

|Model|Train accuracy|Test accuracy|5-fold CV|
|-|-|-|-|
|Random Forest|1.000|0.891|0.875 ± 0.023|
|Logistic Regression|0.907|0.882|0.882 ± 0.014|
|Random Forest (tuned)|n/a|0.886|0.883|

* The Random Forest scores 1.0 on training data but about 0.89 on test data, so it **overfits**. Logistic Regression is nearly as accurate with a much smaller train/test gap.
* Tuning gave no real improvement: the differences are within the noise of cross-validation.
* A test set of 220 students is small, so differences of about 1% are not meaningful.

## What the model relies on

Mean absolute SHAP values put **blood pressure**, **sleep quality**, **teacher-student relationship**, **academic performance** and **basic needs** at the top.

Blood pressure is the strongest signal, but its relationship with stress in this data is not smooth (the average value is higher for Low than for Medium), another sign the data is synthetic. I would not read it as a real medical finding.

## How the app explains results without SHAP

The app does not import `shap`, because it can be blocked or heavy on some machines. Instead, for each answer it tries every value on that question's scale and measures how much the model's confidence changes compared with your actual answer. On the 220 test students, this picked the same top reason as SHAP 90% of the time (top-3 overlap 0.70), and takes about 17 ms per student.

## Limitations

* Synthetic-looking data: it shows patterns in the data, not real causes of stress.
* Model confidence is not the same as how severe someone's stress is.
* Small test set, and a Random Forest that overfits.
* Not validated on real students; not for clinical use.

## Run it locally

```
pip install -r requirements.txt
python -m streamlit run app.py
```

## Files

|File|Purpose|
|-|-|
|`app.py`|Streamlit app|
|`model.pkl`|Trained Random Forest|
|`features.json`|Feature names, ranges and default values|
|`StudySense.ipynb`|Notebook: exploration, training, tuning, SHAP|
|`.streamlit/config.toml`|Light and dark themes|

## Tech

Python, pandas, scikit-learn, SHAP, Streamlit, joblib.
