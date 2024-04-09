# PROJECT2
---

# Mushroom Classification Project

## Overview
In this project, our team explores the fascinating world of mushrooms, focusing on classifying them as edible or poisonous based on their characteristics. Leveraging the UCI machine learning library's dataset, which includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms, we aim to demonstrate the power of machine learning in aiding the identification of potentially dangerous species.

## Dataset
The dataset, derived from "The Audubon Society Field Guide to North American Mushrooms" (1981), encompasses 23 species of gilled mushrooms, providing a robust foundation for our classification tasks.

## Installation
Before running the notebook, ensure you have the following Python libraries installed:
```python
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import seaborn as sns
import os
from IPython.display import Image
import pydotplus
!pip install pydotplus
!pip install plotly
```

## Data Processing
Our data preprocessing steps included label encoding and dropping non-informative columns. A significant effort was dedicated to maintaining a clear association between labels and their meanings, facilitating intuitive understanding and analysis.

## Model and Insights
We employed a RandomForestClassifier to predict the edibility of mushrooms with surprising accuracy. Through our analysis, we uncovered the critical features influencing a mushroom's classification, offering insights into both the natural world and our model's decision-making process.

## Challenges and Future Directions
Our project also discusses the challenges of relying on AI for identifying mushrooms, especially considering the potential dangers of misidentification. We explored the limitations of current AI models and proposed directions for future research, including the development of more robust classification systems and the expansion of our study to include other characteristics like medicinal properties and ecological impacts.

## Conclusion
By achieving a 100% accuracy in classifying mushrooms as edible or poisonous, our project highlights the potential of machine learning in environmental science and public health. However, we also caution against over-reliance on AI without proper validation and emphasize the importance of continued research and development in this field.

## Code Snippet
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb

# Load the dataset
df = pd.read_csv('mushrooms.csv')  # Update this path to your dataset location

# Encode the categorical data
label_encoders = {}
for column in df.columns:
    if df[column].dtype == type(object):  # Encoding if the column is categorical
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# Separate features and target
X = df.drop('class', axis=1)
y = df['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate an XGBoost classifier
model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss')

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Feature importance
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(importance_df)
```

## Team Members
- Matt
- Nate
- Ken
- Evan
- James

Our project bridges the gap between traditional mycology and modern machine learning, aiming to contribute valuable insights to both fields. We invite feedback and collaboration to further our research and impact.

---

This README provides a comprehensive overview of your project, including the objectives, dataset details, installation instructions, data processing steps, model insights, challenges faced, future directions, and a sample code snippet. Adjustments may be necessary based on the actual paths, dependencies, or any additional project components not covered in the document provided.
