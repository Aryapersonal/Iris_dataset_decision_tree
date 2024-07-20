### The Spark Foundation Data Science & Business Analytics Internship @GRIP #JULY24BATCH

#### TASK 06- PREDICTION USING DECISION TREE ALGORITHAM

##### Dataset : Iris dataset

##### Language: Python,Jupyter Notebook
#### Libraries:
- **Pandas:** For data manipulation and analysis.
- **NumPy:** For numerical operations.
- **Matplotlib & Seaborn:** For data visualization.
- **Scikit-learn Libraries:**
  - **LabelEncoder:** For encoding target labels.
  - **train_test_split:** To split the dataset into training and testing sets.
  - **DecisionTreeClassifier:** To build the decision tree model.
  - **classification_report & confusion_matrix:** For evaluating the model.
  - **plot_tree:** For visualizing the decision tree.

### 
<div align ="right">
  
  <a href="https://youtu.be/pqs5u2FKtu8?si=axOgM3aY2t3MTayL">
    <img src ="https://img.shields.io/static/v1?message=Youtube&logo=youtube&label=&color=FF0000&logoColor=white&labelColor=&srtle=for-the-badge" height="25" alt="youtube logo" />
  </a>  
</div> 

### **Overview**
===========

In this project, I conducted a thorough analysis of the Iris dataset with the primary objective of predicting species classifications using the Decision Tree algorithm  & to provide insights into feature importance and model performance using Python (jupyter notebook).You can also check my youtube link attached to this readme file for my explanation.

**Table of Contents**

1. [Overview](#overview)
2. [Installation](#installation)
3. [Features](#features)
4. [Visualizations and Insights](#visualizations-insights)
5. [Conclusion](#conclusion)
6. [Acknowledgments](#acknowledgments)
7. [Author Information](#Author-Information)

## Installation
To run this project,  you will need Python and Jupyter Notebook installed on your system,


**Features**
--------------------
1. **Understand the Dataset:**
   - Explore the structure and characteristics of the Iris dataset, including the relationships between features and the distribution of species.

2. **Prepare Data for Modeling:**
   - Perform necessary preprocessing steps such as encoding categorical variables and splitting the dataset into training and testing sets to ensure the data is suitable for modeling.

3. **Develop Predictive Models:**
   - Train and evaluate Decision Tree models to classify Iris species. This includes assessing model performance through metrics such as confusion matrices and classification reports.

4. **Analyze Feature Importance:**
   - Determine the significance of different features in the classification process to understand which attributes are most influential in predicting species.

5. **Visualize Model Performance:**
   - Use visualization tools to represent the trained decision tree and the relationships between features, providing an intuitive understanding of the model’s decision-making process.
   

**Visualizations and Insights**
------------------------------
1. **Dataset Overview:**
   - The Iris dataset is well-structured with no missing values, allowing for reliable exploratory analysis and modeling.

2. **Exploratory Data Analysis:**
   - Initial EDA, including descriptive statistics and correlation matrices, highlights the relationships between features and their distributions.
   
      ![download](https://github.com/user-attachments/assets/d91292ec-096d-48c5-8901-af09b9198d62)

     
   - The pair plot visually demonstrates how different features interact and overlap between different species.


      ![download](https://github.com/user-attachments/assets/12dad639-f1f9-42c5-b655-0e16f3a95b93)


  

3. **Data Preparation and Preprocessing:**
   - Target labels were successfully encoded, and the dataset was appropriately split into training and testing sets.
   - The preprocessing steps ensured that the data was ready for effective model training and evaluations
  ```Separating target varibale(y) & feature variables (x)
target = iris['species']
data = iris.copy()
data = data.drop('species',axis =1)

x = data #dependent variable

target #independent variable
```

4. **Model Training and Evaluation:**
   - The default Decision Tree Classifier model and an alternative model with entropy criterion were both evaluated.
   - The default model highlighted `petal_width` as the most significant feature, while the entropy-based model further reinforced its importance, suggesting that `petal_width` plays a central role in species classification.

     
     ![image](https://github.com/user-attachments/assets/c63efa5d-e4e1-494d-9b02-04b45aae9c72)


5. **Performance Metrics:**
   - Confusion matrices and classification reports provided insights into the accuracy and reliability of the models.
   - Evaluating these metrics helps in understanding how well the models performed and identifying areas for potential improvement.
   - Confusion matrices code:
```
y_predt = model.predict(x_test)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_predt))
```
   - Classification matrix code:

```
print('Classification report -\n', classification_report(y_test,y_predt))
```

6. **Visualization:**
   - Visualization tools such as pair plots, heatmaps, and decision tree plots offered a clear understanding of the data and model behavior.
   - The decision tree visualization, in particular, provided an interpretable view of the model's decision-making process.
     
     ![download](https://github.com/user-attachments/assets/d31bbd79-2b17-4bc3-8900-ae797c604317)
7. **Accuarcy Score:**
     - The model achieved an accuracy score of 1.0, which indicates that all predictions made by the decision tree classifier on the test dataset were correct.
     - Accuracy code:
```
# Model Accuracy
import sklearn.metrics as sm
print("Accuracy:",sm.accuracy_score(y_test, y_predt))
```



**Conclusion**

In this project, I conducted a thorough analysis of the Iris dataset with the primary objective of predicting species classifications using the Decision Tree algorithm in Python. The project involved several key steps:

1. **Data Exploration and Preparation:**
   - I performed comprehensive exploratory data analysis (EDA) to understand the dataset’s structure, including its features and target variable.
   - The dataset was carefully inspected for missing values, and relevant preprocessing steps were applied, such as label encoding and feature selection.

2. **Visualization:**
   - Various visualization techniques were employed to explore relationships between features, including pair plots and correlation matrices. These visualizations provided a clear understanding of how different features interact and how they contribute to distinguishing between species.

3. **Model Training and Evaluation:**
   - The Decision Tree Classifier was trained and evaluated to predict the species of Iris flowers. Two models were explored: a default Decision Tree and one with an entropy criterion, revealing insights into feature importance and model performance.
   - The models demonstrated how specific features, particularly `petal_width`, play a crucial role in classification, highlighting the model’s ability to leverage these features effectively.

4. **Results and Insights:**
   - The results from the confusion matrices and classification reports provided a comprehensive view of the model's accuracy and effectiveness. The feature importance analysis underscored the key attributes driving the classification decisions.
   - The visual representation of the decision tree facilitated an intuitive understanding of the model's decision-making process.

Overall, the project successfully showcased the effectiveness of the Decision Tree algorithm in making predictions based on the Iris dataset. The findings offer valuable insights into feature importance and model behavior, which can be instrumental for further analysis and applications in classification tasks. This approach not only highlights the power of decision tree models but also emphasizes the significance of data preprocessing and visualization in achieving robust predictive performance.

**Acknowledgments**

Thanks to the contributors of the libraries used in this project: Pandas, NumPy, Matplotlib, and Seaborn.
Thanks to the creators of the Iris dataset for providing the data used in this analysis.
Special thanks to the Spark Foundation to provide me this internship opportunity to showcase my skills in Data cleaning,EDA ,Decision Tree algoritham & forming meaningul insights.

**Author Information**
----------------------
#### ARYA S : www.linkedin.com/in/aryadataanalyst

**Thank You!**
