# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
def load_and_explore_data():
    """Loading the Iris dataset, time to have some fun with this."""
    try:
        # Load the Iris dataset
        iris = load_iris()
        iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        iris_df['species'] = iris.target
        
        # Map target numbers to species names
        iris_df['species'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        
        # Display first few rows
        print("First 5 rows of the dataset:")
        print(iris_df.head())
        print("\n")
        
        # Explore data structure
        print("Dataset information:")
        print(iris_df.info())
        print("\n")
        
        # Check for missing values
        print("Missing values per column:")
        print(iris_df.isnull().sum())
        print("\n")
        
        return iris_df
    
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

# Task 2: Basic Data Analysis
def perform_data_analysis(df):
    """Perform basic statistical analysis on the dataset."""
    if df is None:
        return
        
    print("Basic statistics for numerical columns:")
    print(df.describe())
    print("\n")
    
    # Group by species and calculate mean for each feature
    print("Mean values by species:")
    print(df.groupby('species').mean())
    print("\n")
    
    # Additional analysis: correlation between features
    print("Correlation matrix:")
    print(df.corr(numeric_only=True))
    print("\n")
    
    # Interesting findings
    print("Interesting findings:")
    print("- Setosa has significantly smaller petal dimensions compared to other species")
    print("- Versicolor and virginica have more overlap in their measurements")
    print("- Sepal length and width have the weakest correlation among the features")

# Task 3: Data Visualization
def create_visualizations(df):
    """Create various visualizations of the dataset."""
    if df is None:
        return
        
    plt.figure(figsize=(15, 10))
    
    # 1. Line chart (simulating trends over time)
    plt.subplot(2, 2, 1)
    for species in df['species'].unique():
        species_data = df[df['species'] == species]
        plt.plot(species_data['sepal length (cm)'], label=species)
    plt.title('Sepal Length by Species (Index Order)')
    plt.xlabel('Sample Index')
    plt.ylabel('Sepal Length (cm)')
    plt.legend()
    
    # 2. Bar chart (average petal length per species)
    plt.subplot(2, 2, 2)
    df.groupby('species')['petal length (cm)'].mean().plot(kind='bar', color=['blue', 'orange', 'green'])
    plt.title('Average Petal Length by Species')
    plt.xlabel('Species')
    plt.ylabel('Petal Length (cm)')
    
    # 3. Histogram (distribution of sepal width)
    plt.subplot(2, 2, 3)
    df['sepal width (cm)'].hist(bins=15, edgecolor='black')
    plt.title('Distribution of Sepal Width')
    plt.xlabel('Sepal Width (cm)')
    plt.ylabel('Frequency')
    
    # 4. Scatter plot (sepal length vs petal length)
    plt.subplot(2, 2, 4)
    colors = {'setosa': 'blue', 'versicolor': 'orange', 'virginica': 'green'}
    for species, color in colors.items():
        species_data = df[df['species'] == species]
        plt.scatter(species_data['sepal length (cm)'], species_data['petal length (cm)'], 
                    color=color, label=species)
    plt.title('Sepal Length vs Petal Length')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load and explore data
    iris_df = load_and_explore_data()
    
    # Perform analysis
    if iris_df is not None:
        perform_data_analysis(iris_df)
        create_visualizations(iris_df)
