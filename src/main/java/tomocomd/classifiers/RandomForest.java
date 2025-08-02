package tomocomd.classifiers;

/*
 *   This clss is based on Kweka's RandomForest class.
 *    RandomForest.java
 *
 */

import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomTree;
import weka.core.Capabilities;
import weka.core.Utils;
import weka.core.WekaException;
import weka.gui.ProgrammaticProperty;

public class RandomForest extends Bagging {

  /** for serialization */
  private static final long serialVersionUID = 7514286911168394708L;

  /** True to compute attribute importance */
  protected boolean mComputeAttributeImportance;

  /**
   * Constructor that sets base classifier for bagging to RandomTre and default number of iterations
   * to 100.
   */
  public RandomForest(int seed, int numExecutionSlots, String name) {
    super(seed, numExecutionSlots, name);
    mComputeAttributeImportance = true;
    RandomTree rTree = new RandomTree();
    rTree.setDoNotCheckCapabilities(true);
    super.setClassifier(rTree);
    super.setRepresentCopiesUsingWeights(true);
    setNumIterations(100);
    ((RandomTree) m_Classifier).setComputeImpurityDecreases(mComputeAttributeImportance);
  }

  public void setNumFeatures(int newNumFeatures) {

    ((RandomTree) getClassifier()).setKValue(newNumFeatures);
  }

  /**
   * Returns default capabilities of the base classifier.
   *
   * @return the capabilities of the base classifier
   */
  @Override
  public Capabilities getCapabilities() {

    // Cannot use the main RandomTree object because capabilities checking has
    // been turned off
    // for that object.
    return (new RandomTree()).getCapabilities();
  }

  /**
   * String describing default classifier.
   *
   * @return the default classifier classname
   */
  @Override
  protected String defaultClassifierString() {

    return "weka.classifiers.trees.RandomTree";
  }

  /**
   * This method only accepts RandomTree arguments.
   *
   * @param newClassifier the RandomTree to use.
   * @exception IllegalArgumentException argument is not a RandomTree
   */
  @Override
  @ProgrammaticProperty
  public void setClassifier(Classifier newClassifier) {
    if (!(newClassifier instanceof RandomTree)) {
      throw new IllegalArgumentException(
          "RandomForest: Argument of setClassifier() must be a RandomTree.");
    }
    super.setClassifier(newClassifier);
  }

  /**
   * Get whether to compute and output attribute importance scores
   *
   * @return true if computing attribute importance scores
   */
  public boolean getComputeAttributeImportance() {
    return mComputeAttributeImportance;
  }

  /** Set the number of decimal places. */
  @Override
  public void setNumDecimalPlaces(int num) {

    super.setNumDecimalPlaces(num);
    ((RandomTree) getClassifier()).setNumDecimalPlaces(num);
  }

  public void setBreakTiesRandomly(boolean newBreakTiesRandomly) {

    ((RandomTree) getClassifier()).setBreakTiesRandomly(newBreakTiesRandomly);
  }

  public void setMaxDepth(int value) {
    ((RandomTree) getClassifier()).setMaxDepth(value);
  }

  /**
   * Set the preferred batch size for batch prediction.
   *
   * @param size the batch size to use
   */
  @Override
  public void setBatchSize(String size) {

    super.setBatchSize(size);
    ((RandomTree) getClassifier()).setBatchSize(size);
  }

  /**
   * Sets the seed for the random number generator.
   *
   * @param s the seed to be used
   */
  @Override
  public void setSeed(int s) {

    super.setSeed(s);
    ((RandomTree) getClassifier()).setSeed(s);
  }

  /**
   * Returns description of the bagged classifier.
   *
   * @return description of the bagged classifier as a string
   */
  @Override
  public String toString() {

    if (m_Classifiers == null) {
      return "RandomForest: No model built yet.";
    }
    StringBuilder buffer = new StringBuilder("RandomForest\n\n");
    buffer.append(super.toString());

    if (getComputeAttributeImportance()) {
      try {
        double[] nodeCounts = new double[mData.numAttributes()];
        double[] impurityScores = computeAverageImpurityDecreasePerAttribute(nodeCounts);
        int[] sortedIndices = Utils.sort(impurityScores);
        buffer.append(
            "\n\nAttribute importance based on average impurity decrease "
                + "(and number of nodes using that attribute)\n\n");
        for (int i = sortedIndices.length - 1; i >= 0; i--) {
          int index = sortedIndices[i];
          if (index != mData.classIndex()) {
            buffer
                .append(Utils.doubleToString(impurityScores[index], 10, getNumDecimalPlaces()))
                .append(" (")
                .append(Utils.doubleToString(nodeCounts[index], 6, 0))
                .append(")  ")
                .append(mData.attribute(index).name())
                .append("\n");
          }
        }
      } catch (WekaException ex) {
        // ignore
      }
    }

    return buffer.toString();
  }

  /**
   * Computes the average impurity decrease per attribute over the trees
   *
   * @param nodeCounts an optional array that, if non-null, will hold the count of the number of
   *     nodes at which each attribute was used for splitting
   * @return the average impurity decrease per attribute over the trees
   */
  public double[] computeAverageImpurityDecreasePerAttribute(double[] nodeCounts)
      throws WekaException {

    if (m_Classifiers == null) {
      throw new WekaException("Classifier has not been built yet!");
    }

    if (!getComputeAttributeImportance()) {
      throw new WekaException("Stats for attribute importance have not " + "been collected!");
    }

    double[] impurityDecreases = new double[mData.numAttributes()];
    if (nodeCounts == null) {
      nodeCounts = new double[mData.numAttributes()];
    }
    for (Classifier c : m_Classifiers) {
      double[][] forClassifier = ((RandomTree) c).getImpurityDecreases();
      for (int i = 0; i < mData.numAttributes(); i++) {
        impurityDecreases[i] += forClassifier[i][0];
        nodeCounts[i] += forClassifier[i][1];
      }
    }
    for (int i = 0; i < mData.numAttributes(); i++) {
      if (nodeCounts[i] > 0) {
        impurityDecreases[i] /= nodeCounts[i];
      }
    }

    return impurityDecreases;
  }
}
