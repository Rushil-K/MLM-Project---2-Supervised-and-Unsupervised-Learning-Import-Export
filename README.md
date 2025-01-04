Please View this file in the Google Colab Environment as 3D Visualizations might not be visible in the GitHub Repository.

# MLM-Project---2-Supervised-and-Unsupervised-Learning-Import-Export  
Performed Supervised and Unsupervised learning on the Import/Export Dataset.
- Rushil Kohli

# Machine Learning Project: In-Depth Analysis of Global Import/Export Data

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview

This project delves into a comprehensive analysis of a dataset containing detailed global import and export transactions. The primary objective is to leverage both unsupervised and supervised machine learning techniques to uncover hidden patterns, segment diverse trade activities, and gain actionable insights into critical business factors. This analysis emphasizes clustering and dimensionality reduction methods to effectively navigate and interpret the complex structure inherent in international trade data. The aim is to not only understand the data's intricate structure but also to provide strategic recommendations that can enhance business operations and decision-making.

## Project Goals

This project aims to achieve the following key objectives:

* **Unsupervised Learning for Data Segmentation:** Utilize clustering algorithms to effectively segment the dataset, identifying distinct groupings of trade transactions based on a variety of features.
* **Determine Optimal Clustering:** Empirically identify the most suitable number of clusters that accurately represent the inherent structure of the data.
* **In-Depth Cluster Characterization:** Conduct a thorough analysis of each identified cluster to understand its unique characteristics, such as geographic distribution, prevailing product categories, preferred shipping methods, and dominant payment terms.
* **Supervised Learning for Predictive Modeling:** Demonstrate the application of supervised learning, specifically random forest classification, to build predictive models using the import and export data.
* **Derive Actionable Managerial Insights:** Based on the findings, develop strategic insights and recommendations that can guide business decisions, optimize operational efficiency, and improve customer engagement.

## Dataset

The dataset is a rich source of information about international trade transactions, encompassing a variety of features:

* **Categorical Variables:**
    * `Country`: The country of origin or destination.
    * `Product`: The type of product being traded.
    * `Import_Export`: Indicates whether the transaction is an import or export.
    * `Category`: A broader classification of the products.
    * `Port`: The specific port involved in the transaction.
    * `Shipping_Method`: The mode of transport used for shipping.
    * `Supplier`: The supplier of the goods.
    * `Customer`: The recipient of the goods.
    * `Payment_Terms`: The agreed payment conditions.
* **Numerical Variables:**
    * `Quantity`: The amount of product traded.
    * `Value`: The monetary value of the transaction.
    * `Weight`: The weight of the shipped goods.
    * `Date`: The date of the transaction.
* **Identifiers:**
    * `Transaction_ID`: A unique identifier for each transaction.
    * `Customs_Code`: A unique identifier for each product.
    * `Invoice_Number`: A unique identifier for each invoice.

**Important Note:** The dataset was subjected to rigorous preprocessing steps, including data cleaning, feature engineering, and categorical encoding, to prepare it for advanced machine learning algorithms. Key identifiers such as `Transaction_ID`, `Customs_Code`, and `Invoice_Number` were excluded from the clustering analysis as they were unique to each record and added no clustering value.

## Methodology

### Data Preprocessing

1. **Data Loading and Cleaning:** The dataset, sourced from a Google Drive URL, was initially loaded and thoroughly checked for missing or inconsistent values.
2. **Feature Engineering:** The `Date` variable was transformed into `Year`, `Month`, and `Day` columns to capture potential temporal patterns.
3. **Categorical Encoding:** Categorical variables were encoded using `OrdinalEncoder` to convert them into a numerical format suitable for machine learning models.
4. **Scaling:** Numerical data underwent scaling using both `StandardScaler` and `MinMaxScaler` to normalize the data and ensure uniformity across features.

### Unsupervised Machine Learning

1. **K-Means Clustering:**
    * **Optimal K Determination:** The Elbow Method and silhouette scores were employed to identify the optimal number of clusters (`k`) that best represents the data's structure.
    * **Iterative Clustering:** K-Means was applied iteratively with different values of `k` to evaluate cluster performance and stability.
    * **Initial vs. Refined Clusters:** An initial `k=2` was used, but later adjusted to `k=6` based on insights from PCA visualizations, revealing more nuanced groupings.
    * **Multiple K:** The number of clusters was later increased to 10 and 15.
2. **Dimensionality Reduction (PCA):**
    * **Visualization Aid:** Principal Component Analysis (PCA) was used to reduce the data's dimensionality, making it possible to visualize the clusters in 2D and 3D spaces.
    * **Hidden Structure Detection:** PCA helped in revealing potential hidden cluster structures not immediately apparent to K-Means.
    * **Variance Exploration:** The variance explained by each PCA component was assessed to understand the relative importance of each dimension.
3. **Advanced Clustering Techniques:**
    * **DBSCAN:** Density-Based Spatial Clustering of Applications with Noise (DBSCAN) was applied to identify clusters based on density rather than distance.
    * **Agglomerative Clustering:** Hierarchical agglomerative clustering was employed to group similar data points together in a hierarchical manner.
4. **UMAP:**
    * The data was processed using UMAP.
5. **Non-linear Dimensionality Reduction (t-SNE):**
    * t-Distributed Stochastic Neighbor Embedding (t-SNE) was used for non-linear dimensionality reduction, providing another perspective on the data's structure.

### Supervised Machine Learning

1. **Logistic Regression:**
    * **Model Overview:** Logistic Regression was applied to predict the likelihood of a transaction being an import or export based on various features. It is a linear model, ideal for binary classification tasks.
    * **Training and Evaluation:** The model was trained on the data and evaluated using standard metrics such as accuracy, precision, recall, and F1-score.
  
2. **Decision Tree:**
    * **Model Overview:** A Decision Tree model was used for classification, helping to visualize how decisions are made based on the feature values.
    * **Training and Evaluation:** The tree was trained on the dataset, and its performance was evaluated using metrics like accuracy and confusion matrix.

3. **Random Forest:**
    * **Model Overview:** Random Forest is an ensemble method that combines multiple decision trees to improve prediction accuracy and robustness.
    * **Training and Evaluation:** A Random Forest classifier was trained and evaluated, with feature importance analysis to identify the most influential features for classification.

4. **XGBoost:**
    * **Model Overview:** XGBoost is a gradient boosting framework that builds strong predictive models by combining weak models (decision trees) sequentially.
    * **Training and Evaluation:** The XGBoost model was applied to the dataset, and its performance was assessed using the evaluation metrics. The model often outperforms other algorithms due to its efficiency and accuracy.

5. **Gradient Boosting:**
    * **Model Overview:** Gradient Boosting builds models sequentially, each focusing on the errors of the previous model, making it particularly effective in learning from residuals.
    * **Training and Evaluation:** Gradient Boosting was used to predict whether a transaction was an import or export, and the model’s performance was evaluated using the same classification metrics.

6. **Trend Prediction and Visualization:**
    * **Model Output Visualization:** To predict trends over time, line graphs were used to visualize the predicted outcomes from the models. These graphs helped illustrate how the predicted probability of import/export transactions changes over time, showing seasonal trends and fluctuations in trade activities.
    * **Trend Analysis:** By plotting the predictions from all the models, you could observe the patterns over different time periods (such as monthly or quarterly), allowing for better forecasting and understanding of trends in international trade.

## Key Findings

* **Cluster Diversity and Complexity:** The analysis revealed a diverse set of clusters, each with unique characteristics, suggesting varying trade patterns and behaviors.
* **Geographic and Product Segmentation:** Clusters showed clear tendencies toward specific geographic regions and product categories, indicating potential market specializations.
* **Shipping and Payment Behavior Insights:** Distinct preferences for shipping methods and payment terms were observed across different clusters, highlighting the existence of diverse market segments.
* **Data High-Dimensionality:** The data exhibits a complex, high-dimensional structure, with gradual transitions between clusters, rather than sharp boundaries.
* **PCA Limitations and Hidden Structures:** While PCA was useful for visualization, it also indicated the presence of more complex, hidden structures that were not fully captured by K-Means alone.
* **UMAP Advantage:** UMAP provided better dimensionality reduction than PCA and T-SNE, in particular with high-dimensional data.
* **Feature Importance:** Supervised learning revealed the key features that are most predictive of whether a transaction is an import or export, offering deeper insights into trade dynamics.
* **Nested Structures:** The data seems to have multiple levels of clusters.
* **Gradual Transitions:** The cluster boundaries are gradual and not sharp.

## Managerial Insights

* **Trade Flow Analysis:** Identified import-focused vs. export-focused clusters, revealing regional specializations and trade imbalances.
* **Payment Behavior Dynamics:** Analyzed cluster-specific payment terms, providing valuable information for cash flow optimization and risk management.
* **Shipping Efficiency Strategies:** Identified shipping method preferences within clusters, aiding in logistical planning and cost reduction.
* **Strategic Growth Opportunities:** Uncovered potential to target untapped markets and optimize operations across various market segments.
* **Customer Behavior Patterns:** Identified behavioral patterns

 within clusters, which can inform tailored marketing and loyalty strategies.

## Next Steps

* **In-Depth Cluster Investigation:** Conduct a more detailed exploration of smaller clusters to pinpoint niche market opportunities.
* **Time-Series Forecasting:** Implement time-series forecasting techniques to better manage seasonal trends and optimize inventory and pricing strategies.
* **Supplier and Customer Optimization:** Refine supplier contracts and tailor marketing approaches based on specific customer behaviors identified within clusters.
* **Further Analysis with UMAP:** Explore the data further using UMAP, leveraging its non-linear properties.

## Libraries Used

* **Core Libraries:** `numpy` (version 1.26.4), `pandas` (version 2.2.0)
* **Data Visualization:** `matplotlib` (version 3.8.2), `seaborn` (version 0.13.2), `plotly` (version 5.18.0)
* **Unsupervised Learning:**
    * `sklearn.cluster` (KMeans, DBSCAN, AgglomerativeClustering) from scikit-learn (version 1.3.2)
    * `sklearn.decomposition` (PCA) from scikit-learn (version 1.3.2)
    * `sklearn.preprocessing` (StandardScaler, MinMaxScaler, OrdinalEncoder) from scikit-learn (version 1.3.2)
    * `sklearn.metrics` from scikit-learn (version 1.3.2)
    * `umap.umap_` (version 0.5.5)
* **Dimensionality Reduction:** `sklearn.manifold` (TSNE) from scikit-learn (version 1.3.2)
* **Supervised Learning:**
    * `sklearn.model_selection` (train_test_split) from scikit-learn (version 1.3.2)
    * `sklearn.ensemble` (RandomForestClassifier) from scikit-learn (version 1.3.2)
* **Other:** `requests` (version 2.31.0), `io`

## License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

---

# Follow Me on GitHub: https://github.com/Rushil-K

