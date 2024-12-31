package tomocomd.utils;

import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import tomocomd.ModelingException;
import weka.core.Instances;

public class PearsonCorrelationBetweenAttributes {

  /**
   * Computes the Pearson correlation coefficient between two attributes in the dataset using Apache
   * Commons Math3.
   *
   * @param data The dataset (Instances object).
   * @param attrIndex1 Index of the first attribute.
   * @param attrIndex2 Index of the second attribute.
   * @return The Pearson correlation coefficient.
   */
  public static double computeCorrelation(Instances data, int attrIndex1, int attrIndex2) {
    if (data.attribute(attrIndex1).isNominal() || data.attribute(attrIndex2).isNominal()) {
      throw ModelingException.ExceptionType.FILTERING_EXCEPTION.get(
          "Attributes must be numeric for compute correlation.");
    }

    double[] values1 = data.attributeToDoubleArray(attrIndex1);
    double[] values2 = data.attributeToDoubleArray(attrIndex2);

    // Handle missing values by filtering them out
    int n = values1.length;
    double[] filteredValues1 = new double[n];
    double[] filteredValues2 = new double[n];
    int validCount = 0;

    for (int i = 0; i < n; i++) {
      if (!Double.isNaN(values1[i]) && !Double.isNaN(values2[i])) {
        filteredValues1[validCount] = values1[i];
        filteredValues2[validCount] = values2[i];
        validCount++;
      }
    }

    double[] finalValues1 = new double[validCount];
    double[] finalValues2 = new double[validCount];
    System.arraycopy(filteredValues1, 0, finalValues1, 0, validCount);
    System.arraycopy(filteredValues2, 0, finalValues2, 0, validCount);

    PearsonsCorrelation pearsonsCorrelation = new PearsonsCorrelation();
    return pearsonsCorrelation.correlation(finalValues1, finalValues2);
  }

  /**
   * Computes a lower triangular matrix of correlations for all numeric attributes in the dataset.
   *
   * @param data The dataset (Instances object).
   * @return A 2D array representing the lower triangular matrix of correlations.
   */
  public static TriangularMatrixAsVector computeCorrelationMatrix(Instances data) {
    int numAttributes = data.numAttributes();
    TriangularMatrixAsVector correlationMatrix = new TriangularMatrixAsVector(numAttributes);

    for (int i = 0; i < numAttributes; i++) {
      if (i == data.classIndex()) continue;
      for (int j = 0; j <= i; j++) {
        if (j == data.classIndex()) continue;
        if (i == j) {
          correlationMatrix.setEntry(i, j, 1.0); // Correlation of an attribute with itself is 1
        } else {
          correlationMatrix.setEntry(i, j, computeCorrelation(data, i, j));
        }
      }
    }

    return correlationMatrix;
  }
}
