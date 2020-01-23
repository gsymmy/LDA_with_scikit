import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda

names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Class Label']
df = pd.io.parsers.read_csv("iris.data")
df.columns = names
df.head()

features = df.drop('Class Label', axis=1)
classlabels = df['Class Label']

sklearn_lda = lda(n_components=2)
sklearn_lda_features = sklearn_lda.fit_transform(features, classlabels)

def plot_lda(two_lda_dimensions, label_matrix, title):
    
    # Make scatter plot, with labels and colors
    for label,marker,color in zip(
        ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),('^', 's', 'o'),('orange', 'black', 'grey')):

        plt.scatter(x=two_lda_dimensions[:,0][label_matrix == label],
                    y=two_lda_dimensions[:,1][label_matrix == label],
                    marker=marker,
                    color=color,
                    label=label)
    
    # Label axes
    plt.xlabel('Linear Discriminant 1')
    plt.ylabel('Linear Discriminant 2')
    
    # Add plot legend
    leg = plt.legend(loc='best', fancybox=True)
    
    # Add plot title
    plt.title(title)

plot_lda(two_lda_dimensions=sklearn_lda_features,
                 label_matrix=classlabels,
                 title='LDA via scikit-learn')

plt.show()

def plot_lda_decision_boundaries(two_lda_dimensions, label_matrix):
    
    # Create mesh_matrix, a mesh of points in space of first two linear discriminants
    ldone_min, ldone_max = two_lda_dimensions[:,0].min() - 1, two_lda_dimensions[:,0].max() + 1
    ldtwo_min, ldtwo_max = two_lda_dimensions[:,1].min() - 1, two_lda_dimensions[:,1].max() + 1

    ldoneone, ldtwotwo = np.meshgrid(np.linspace(ldone_min, ldone_max, 500),
                                 np.linspace(ldtwo_min, ldtwo_max, 500))

    mesh_matrix = np.c_[ldoneone.ravel(), ldtwotwo.ravel()]
    
    # Instantiate LDA model and fit LDA model on two_lda_dimensions
    lda_model = lda(n_components=2)
    lda_model.fit(two_lda_dimensions, label_matrix)
                                                                                   
    # Use LDA model to make categorical predictions on mesh_matrix
    mesh_predictions = lda_model.predict(mesh_matrix)
    
    # Map categorical predictions into numerical values for contour plotting
    speciesmap = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
    f = lambda x: speciesmap[x]
    fv = np.vectorize(f)
    Z = fv(mesh_predictions).reshape((len(ldoneone), len(ldtwotwo)))
    
    # Make contour plot
    plt.contourf(ldoneone, ldtwotwo, Z, levels=[-0.5,0.5,1.5,2.5], colors=('orange', 'black', 'grey'), alpha=0.4)

# Plot first two LDA dimensions along with decision boundaries

plot_lda(two_lda_dimensions=sklearn_lda_features,
        label_matrix=classlabels,
        title='Scikit-learn LDA With Decision Boundaries')

plot_lda_decision_boundaries(two_lda_dimensions=sklearn_lda_features,
                             label_matrix=classlabels)

plt.show()



