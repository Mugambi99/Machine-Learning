# K-means Clustering on Mall Customer Data

This repository contains code and analysis for performing K-means clustering on the Mall Customer dataset. The dataset provides information about customers' spending habits and demographic characteristics.

## Dataset

The Mall Customer dataset consists of the following features:

- CustomerID: Unique identifier for each customer
- Gender: Gender of the customer (Male or Female)
- Age: Age of the customer
- Annual Income: Annual income of the customer in dollars
- Spending Score: Score assigned to the customer based on their spending habits and behavior

The dataset is available in the `mall_customers.csv` file.

## Dependencies

The following dependencies are required to run the code:

- Python 3
- pandas
- numpy
- scikit-learn
- matplotlib


## Code

The main code for performing K-means clustering on the dataset is available in the `kmeans_clustering.ipynb` Jupyter Notebook. It covers the following steps:

1. Loading and exploring the dataset
2. Preprocessing the data (if required)
3. Feature selection (if required)
4. Scaling the features (if required)
5. Determining the optimal number of clusters using the elbow method
6. Training the K-means model
7. Visualizing the clusters

Feel free to modify the code to suit your specific needs.

## Results

The results of the K-means clustering analysis are presented in the notebook. The key outputs include:

- Elbow plot showing the within-cluster sum of squares (WCSS) for different numbers of clusters
- Visualization of the clusters and cluster centers

The results can be used to gain insights into customer segmentation based on spending habits and demographic characteristics.

## Acknowledgments

- The Mall Customer dataset is sourced from [Kaggle](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python).
- This project is for educational purposes and can be used as a starting point for further analysis.

## License

This project is licensed under the [MIT License](LICENSE).




