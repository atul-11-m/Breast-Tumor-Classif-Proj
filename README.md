# Breast-Tumor-Classif-Proj

Breast Cancer Detection Model
Problem:
According to the National Breast Cancer Foundation, 1 in 8 women in the United States will be diagnosed with breast cancer in their lifetime. Breast cancer is the second leading cause of cancer death in U.S. women, and it’s estimated that in 2026 alone, over 40,000 women in the country will die from breast cancer. As breast cancer rates have continued to rise over the past decade, it is important to raise awareness for early detection and the use of advanced technologies to help accurately detect cancerous tumors in the early stages, so treatment can be started as soon as possible, increasing a patient's chances of survival.  Biopsies are the most common way of detecting whether or not a tumor is cancerous or not. In this procedure, tissue from the tumor is removed and analyzed in a lab to see if the tissue contains cancer cells. However, there are some potential downsides to this procedure, such as being time-consuming, expensive, and unnecessary if the tumor is found to be benign or not cancerous.  Machine learning models can help in detecting whether or not a tumor found is cancerous or not by analyzing a tumor's size, shape, cell texture, radius, and perimeter measurements, and imaging patterns.

Objective: 
Our goal is to develop and train a machine learning model to classify whether a tumor found in the breast tissue is malignant (cancerous) or benign (non-cancerous) at an early stage, so patients can receive treatment sooner and improve their chances of survival. 

Strategic Aspects: 
Data-Driven Medical Decision Making:
This project aims to demonstrate how data science and machine learning can assist in medical decision-making. By analyzing diagnostic tumor characteristics such as radius, texture, perimeter, and cell symmetry, the model can identify patterns associated with malignant and benign tumors. This allows healthcare professionals to use data-driven insights when evaluating tumor characteristics.

Improving Diagnostic Accuracy:
Traditional diagnostic methods such as biopsies and imaging analysis rely heavily on manual interpretation by medical professionals. While these techniques are effective, they can still lead to misdiagnosis or unnecessary procedures. Machine learning models can analyze many features simultaneously and detect complex patterns in tumor data, potentially improving the accuracy of tumor classification.

Early Detection and Patient Outcomes:
One of the most important strategic aspects of this project is its potential to support early cancer detection. Early identification of malignant tumors can significantly improve treatment effectiveness and patient survival rates. By building a model that classifies tumors at an early stage, this project highlights how technology can help support faster medical intervention.

Application of Machine Learning in Healthcare:
Machine learning is becoming increasingly important in healthcare for tasks such as disease prediction, medical imaging analysis, and diagnostic support. This project demonstrates how classification models can be applied to medical datasets to solve real-world healthcare problems. By training and evaluating different machine learning algorithms, the project explores how these models can be used to analyze medical data and assist doctors in diagnosing breast cancer more efficiently.

Dataset:
The model is going to be trained using the Breast Cancer Wisconsin (Diagnostic) dataset taken from the UC Irvine Machine Learning Repository. This dataset includes 569 tumor samples, 30 different features describing tumor cell characteristics, and binary classification labels that indicate whether or not the tumor is malignant or benign. Features stated in the dataset include: 
radius (average distance from center to tumor boundary)
texture (variation in pixel intensity)
perimeter
area
smoothness
concavity
symmetry

Plan:
We are going to use the Breast Cancer Wisconsin (Diagnostic) dataset to build a machine learning model that can classify tumors as malignant or benign. First, we will load and clean the dataset in Python by removing unnecessary information and preparing the diagnosis labels for analysis. Then, we will explore the dataset to understand the tumor features and identify patterns between malignant and benign tumors. Next, we will split the data into training and testing sets and train multiple machine learning models, including Logistic Regression, Random Forest, and Support Vector Machine. We will compare the models using evaluation methods such as accuracy, precision, and recall matrices to determine which model performs best in detecting tumors. Finally, we will use Matplotlib and Tableau to create visualizations and dashboards that show feature patterns, tumor distributions, and model performance. This will allow us to clearly present our findings and show how machine learning can help classify tumors earlier and more accurately.

Tools:
The project will be implemented using Python and commonly used data science libraries, including:
Python – Used to build and train the machine learning model
Pandas – Used to load and clean the dataset.
NumPy – Used for numerical calculations.
Scikit-learn – Used to train and test machine learning models.
Matplotlib – Used to create graphs and visualizations.
Tableau – Used to create visualizations and dashboards to present the data and results.

These tools will allow for efficient data analysis, model training, and visualization of results.

Group Members:
Atul Marichetty - am3954
Pranav Dharayan - pd610

Citations:
“Breast Cancer Facts & Stats 2026 - Incidence, Age, Survival, & More.” National Breast Cancer Foundation, 3 Mar. 2026, www.nationalbreastcancer.org/breast-cancer-facts/. 
Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B  
