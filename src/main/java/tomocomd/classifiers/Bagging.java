package tomocomd.classifiers;

/*
@   Author Luis
 *    This is version of Weka Bagging.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import tomocomd.utils.ModelingException;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.*;

public class Bagging extends RandomizableParallelEnhancer
    implements WeightedInstancesHandler,
        AdditionalMeasureProducer,
        TechnicalInformationHandler,
        PartitionGenerator,
        Aggregateable<Bagging> {

  /** for serialization */
  private static final long serialVersionUID = -371997031158799622L;

  private static final Logger logger = LogManager.getLogger(Bagging.class);

  protected int mBagSizePercent;
  protected boolean mCalcOutOfBag;
  protected boolean mRepresentUsingWeights;
  protected Evaluation mOutOfBagEvaluationObject;
  private boolean mStoreOutOfBagPredictions;
  private boolean mOutputOutOfBagComplexityStatistics;
  private boolean mNumeric;
  private boolean mPrintClassifiers;
  protected Random mRandom;
  protected boolean[][] mInBag;
  protected Instances mData;

  /** Constructor. */
  protected Bagging() {
    super(1, 1, "Bagging");
  }

  public Bagging(int seed, int numExecutionSlots, String name) {
    super(seed, numExecutionSlots, name);
    mBagSizePercent = 100;
    mCalcOutOfBag = false;
    mRepresentUsingWeights = false;
    mOutOfBagEvaluationObject = null;
    mStoreOutOfBagPredictions = false;
    mOutputOutOfBagComplexityStatistics = false;
    mNumeric = false;
    mPrintClassifiers = false;
    mRandom = null;
    mInBag = null;
    mData = null;
    m_Classifier = new weka.classifiers.trees.REPTree();
  }

  public void setStoreOutOfBagPredictions(boolean storeOutOfBag) {
    mStoreOutOfBagPredictions = storeOutOfBag;
  }

  public void setOutputOutOfBagComplexityStatistics(boolean b) {

    mOutputOutOfBagComplexityStatistics = b;
  }

  public void setRepresentCopiesUsingWeights(boolean representUsingWeights) {

    mRepresentUsingWeights = representUsingWeights;
  }

  public void setBagSizePercent(int newBagSizePercent) {

    mBagSizePercent = newBagSizePercent;
  }

  public void setCalcOutOfBag(boolean calcOutOfBag) {

    mCalcOutOfBag = calcOutOfBag;
  }

  public int getBagSizePercent() {

    return mBagSizePercent;
  }

  /**
   * String describing default classifier.
   *
   * @return the default classifier classname
   */
  @Override
  protected String defaultClassifierString() {
    return "weka.classifiers.trees.REPTree";
  }

  /**
   * Get whether copies of instances are represented using weights rather than explicitly.
   *
   * @return whether copies of instances are represented using weights rather than explicitly
   */
  public boolean getRepresentCopiesUsingWeights() {

    return mRepresentUsingWeights;
  }

  /**
   * Get whether the out of bag predictions are stored.
   *
   * @return whether the out of bag predictions are stored
   */
  public boolean getStoreOutOfBagPredictions() {

    return mStoreOutOfBagPredictions;
  }

  /**
   * Get whether the out of bag error is calculated.
   *
   * @return whether the out of bag error is calculated
   */
  public boolean getCalcOutOfBag() {

    return mCalcOutOfBag;
  }

  /**
   * Gets whether complexity statistics are output when OOB estimation is performed.
   *
   * @return whether statistics are calculated
   */
  public boolean getOutputOutOfBagComplexityStatistics() {

    return mOutputOutOfBagComplexityStatistics;
  }

  /**
   * Get whether to print the individual ensemble classifiers in the output
   *
   * @return true if the individual classifiers are to be printed
   */
  public boolean getPrintClassifiers() {
    return mPrintClassifiers;
  }

  /**
   * Gets the out of bag error that was calculated as the classifier was built. Returns error rate
   * in classification case and mean absolute error in regression case.
   *
   * @return the out of bag error; -1 if out-of-bag-error has not been estimated
   */
  public double measureOutOfBagError() {

    if (mOutOfBagEvaluationObject == null) {
      return -1;
    }
    if (mNumeric) {
      return mOutOfBagEvaluationObject.meanAbsoluteError();
    } else {
      return mOutOfBagEvaluationObject.errorRate();
    }
  }

  /**
   * Returns an enumeration of the additional measure names.
   *
   * @return an enumeration of the measure names
   */
  @Override
  public Enumeration<String> enumerateMeasures() {

    Vector<String> newVector = new Vector<>(1);
    newVector.addElement("measureOutOfBagError");
    return newVector.elements();
  }

  /**
   * Returns the value of the named measure.
   *
   * @param additionalMeasureName the name of the measure to query for its value
   * @return the value of the named measure
   * @throws IllegalArgumentException if the named measure is not supported
   */
  @Override
  public double getMeasure(String additionalMeasureName) {

    if (additionalMeasureName.equalsIgnoreCase("measureOutOfBagError")) {
      return measureOutOfBagError();
    } else {
      throw new IllegalArgumentException(additionalMeasureName + " not supported (Bagging)");
    }
  }

  /**
   * Returns a training set for a particular iteration.
   *
   * @param iteration the number of the iteration for the requested training set.
   * @return the training set for the supplied iteration number
   * @throws ModelingException if something goes wrong when generating a training set.
   */
  @Override
  protected synchronized Instances getTrainingSet(int iteration) throws ModelingException {

    Random r = new Random((long) mSeed + iteration);

    // create the in-bag indicator array if necessary
    if (mCalcOutOfBag) {
      mInBag[iteration] = new boolean[mData.numInstances()];
      return mData.resampleWithWeights(
          r, mInBag[iteration], getRepresentCopiesUsingWeights(), mBagSizePercent);
    } else {
      return mData.resampleWithWeights(r, null, getRepresentCopiesUsingWeights(), mBagSizePercent);
    }
  }

  private void validateInput(Instances data) throws Exception {
    // can classifier handle the data?
    getCapabilities().testWithFail(data);

    // Has user asked to represent copies using weights?
    if (getRepresentCopiesUsingWeights() && !(m_Classifier instanceof WeightedInstancesHandler)) {
      throw new IllegalArgumentException(
          "Cannot represent copies using weights when "
              + "base learner in bagging does not implement "
              + "WeightedInstancesHandler.");
    }
  }

  /**
   * Bagging method.
   *
   * @param data the training data to be used for generating the bagged classifier.
   * @throws Exception if the classifier could not be built successfully
   */
  @Override
  public void buildClassifier(Instances data) throws Exception {

    validateInput(data);

    // get fresh Instances object
    mData = new Instances(data);

    super.buildClassifier(mData);

    mRandom = new Random(mSeed);

    mInBag = null;
    if (mCalcOutOfBag) mInBag = new boolean[m_Classifiers.length][];

    for (Classifier mClassifier : m_Classifiers) {
      if (m_Classifier instanceof Randomizable) {
        ((Randomizable) mClassifier).setSeed(mRandom.nextInt());
      }
    }

    mNumeric = mData.classAttribute().isNumeric();

    buildClassifiers();

    // calc OOB error?
    if (getCalcOutOfBag()) {
      mOutOfBagEvaluationObject = new Evaluation(mData);

      for (int i = 0; i < mData.numInstances(); i++) {
        double[] votes;
        if (mNumeric) votes = new double[1];
        else votes = new double[mData.numClasses()];

        // determine predictions for instance
        int voteCount = 0;
        for (int j = 0; j < m_Classifiers.length; j++) {
          if (mInBag[j][i]) continue;

          if (mNumeric) {
            double pred = m_Classifiers[j].classifyInstance(mData.instance(i));
            if (!Utils.isMissingValue(pred)) {
              votes[0] += pred;
              voteCount++;
            }
          } else {
            voteCount++;
            double[] newProbs = m_Classifiers[j].distributionForInstance(mData.instance(i));
            // sum the probability estimates
            for (int k = 0; k < newProbs.length; k++) {
              votes[k] += newProbs[k];
            }
          }
        }

        // "vote"
        if (mNumeric) {
          if (voteCount > 0) {
            votes[0] /= voteCount;
            mOutOfBagEvaluationObject.evaluationForSingleInstance(
                votes, mData.instance(i), getStoreOutOfBagPredictions());
          }
        } else {
          double sum = Utils.sum(votes);
          if (sum > 0) {
            Utils.normalize(votes, sum);
            mOutOfBagEvaluationObject.evaluationForSingleInstance(
                votes, mData.instance(i), getStoreOutOfBagPredictions());
          }
        }
      }
    } else {
      mOutOfBagEvaluationObject = null;
    }

    // save memory
    mInBag = null;
    mData = new Instances(mData, 0);
  }

  /**
   * Calculates the class membership probabilities for the given test instance.
   *
   * @param instance the instance to be classified
   * @return preedicted class probability distribution
   * @throws Exception if distribution can't be computed successfully
   */
  @Override
  public double[] distributionForInstance(Instance instance) throws Exception {

    double[] sums = new double[instance.numClasses()];
    double[] newProbs;

    double numPreds = 0;
    for (int i = 0; i < m_NumIterations; i++) {
      if (mNumeric) {
        double pred = m_Classifiers[i].classifyInstance(instance);
        if (!Utils.isMissingValue(pred)) {
          sums[0] += pred;
          numPreds++;
        }
      } else {
        newProbs = m_Classifiers[i].distributionForInstance(instance);
        for (int j = 0; j < newProbs.length; j++) {
          sums[j] += newProbs[j];
        }
      }
    }
    if (mNumeric) {
      if (numPreds == 0) {
        sums[0] = Utils.missingValue();
      } else {
        sums[0] /= numPreds;
      }
      return sums;
    }
    if (Utils.eq(Utils.sum(sums), 0)) {
      return sums;
    }
    Utils.normalize(sums);
    return sums;
  }

  /**
   * Set the batch size to use. Gets passed through to the base learner if it implements
   * BatchPredictor. Otherwise, it is just ignored.
   *
   * @param size the batch size to use
   */
  @Override
  public void setBatchSize(String size) {

    if (getClassifier() instanceof BatchPredictor) {
      ((BatchPredictor) getClassifier()).setBatchSize(size);
    } else {
      super.setBatchSize(size);
    }
  }

  /**
   * Gets the preferred batch size from the base learner if it implements BatchPredictor. Returns 1
   * as the preferred batch size otherwise.
   *
   * @return the batch size to use
   */
  @Override
  public String getBatchSize() {

    if (getClassifier() instanceof BatchPredictor) {
      return ((BatchPredictor) getClassifier()).getBatchSize();
    } else {
      return super.getBatchSize();
    }
  }

  /**
   * Batch scoring method. Calls the appropriate method for the base learner if it implements
   * BatchPredictor. Otherwise, it simply calls the distributionForInstance() method repeatedly.
   *
   * @param insts the instances to get predictions for
   * @return an array of probability distributions, one for each instance
   * @throws Exception if a problem occurs
   */
  @Override
  public double[][] distributionsForInstances(Instances insts) throws Exception {

    if (getClassifier() instanceof BatchPredictor) {
      return distributionsForInstancesForBatchPredictor(insts);
    } else {

      double[][] result = new double[insts.numInstances()][insts.numClasses()];
      for (int i = 0; i < insts.numInstances(); i++) {
        result[i] = distributionForInstance(insts.instance(i));
      }
      return result;
    }
  }

  private double[][] distributionsForInstancesForBatchPredictor(Instances insts) {

    ExecutorService pool = Executors.newFixedThreadPool(mNumExecutionSlots);

    // Set up result set, and chunk size
    final int chunksize = m_Classifiers.length / mNumExecutionSlots;
    Set<Future<double[][]>> results = new HashSet<>();

    // For each thread
    for (int j = 0; j < mNumExecutionSlots; j++) {

      // Determine batch to be processed
      final int lo = j * chunksize;
      final int hi = (j < mNumExecutionSlots - 1) ? (lo + chunksize) : m_Classifiers.length;

      // Create and submit new job for each batch of instances
      Future<double[][]> futureT = pool.submit(() -> createJob(insts, lo, hi));
      results.add(futureT);
    }

    // Form ensemble prediction
    double[][] ensemblePreds =
        new double[insts.numInstances()]
            [insts.classAttribute().isNumeric() ? 2 : insts.numClasses()];
    try {
      for (Future<double[][]> futureT : results) {
        double[][] preds = futureT.get();
        for (int j = 0; j < preds.length; j++) {
          for (int k = 0; k < preds[j].length; k++) {
            ensemblePreds[j][k] += preds[j][k];
          }
        }
      }
    } catch (Exception e) {
      logger.error("RandomCommittee: predictions could not be generated by thread.", e);
      Thread.currentThread().interrupt();
    }
    pool.shutdown();

    // Normalise ensemble predictions
    return normaliceEnsemblePredictionsForDistributionsForInstancesForBatchPredictor(
        insts, ensemblePreds);
  }

  private double[][] normaliceEnsemblePredictionsForDistributionsForInstancesForBatchPredictor(
      Instances insts, double[][] ensemblePreds) {
    if (insts.classAttribute().isNumeric()) {
      double[][] finalPreds = new double[ensemblePreds.length][1];
      for (int j = 0; j < ensemblePreds.length; j++) {
        if (ensemblePreds[j][1] == 0) {
          finalPreds[j][0] = Utils.missingValue();
        } else {
          finalPreds[j][0] = ensemblePreds[j][0] / ensemblePreds[j][1];
        }
      }
      return finalPreds;
    } else {
      for (double[] ensemblePred : ensemblePreds) {
        double sum = Utils.sum(ensemblePred);
        if (!Utils.eq((sum), 0)) {
          Utils.normalize(ensemblePred, sum);
        }
      }
      return ensemblePreds;
    }
  }

  private double[][] createJob(Instances insts, int lo, int hi) throws Exception {
    if (insts.classAttribute().isNumeric()) {
      return createNumericJob(insts, lo, hi);
    }
    double[][] ensemblePreds = new double[insts.numInstances()][insts.numClasses()];
    for (int i = lo; i < hi; i++) {
      double[][] preds = ((BatchPredictor) m_Classifiers[i]).distributionsForInstances(insts);
      for (int j = 0; j < preds.length; j++) {
        for (int k = 0; k < preds[j].length; k++) {
          ensemblePreds[j][k] += preds[j][k];
        }
      }
    }
    return ensemblePreds;
  }

  private double[][] createNumericJob(Instances insts, int lo, int hi) throws Exception {

    double[][] ensemblePreds = new double[insts.numInstances()][2];
    for (int i = lo; i < hi; i++) {
      double[][] preds = ((BatchPredictor) m_Classifiers[i]).distributionsForInstances(insts);
      for (int j = 0; j < preds.length; j++) {
        if (!Utils.isMissingValue(preds[j][0])) {
          ensemblePreds[j][0] += preds[j][0];
          ensemblePreds[j][1]++;
        }
      }
    }
    return ensemblePreds;
  }

  /**
   * Returns true if the base classifier implements BatchPredictor and is able to generate batch
   * predictions efficiently
   *
   * @return true if the base classifier can generate batch predictions efficiently
   */
  @Override
  public boolean implementsMoreEfficientBatchPrediction() {
    if (!(getClassifier() instanceof BatchPredictor)) {
      return super.implementsMoreEfficientBatchPrediction();
    }
    return ((BatchPredictor) getClassifier()).implementsMoreEfficientBatchPrediction();
  }

  /**
   * Returns description of the bagged classifier.
   *
   * @return description of the bagged classifier as a string
   */
  @Override
  public String toString() {

    if (m_Classifiers == null) {
      return "Bagging: No model built yet.";
    }
    StringBuffer text = new StringBuffer();
    text.append("Bagging with ")
        .append(getNumIterations())
        .append(" iterations and base learner\n\n")
        .append(getClassifierSpec());
    if (getPrintClassifiers()) {
      text.append("All the base classifiers: \n\n");
      for (Classifier mClassifier : m_Classifiers)
        text.append(mClassifier.toString()).append("\n\n");
    }
    if (mCalcOutOfBag) {
      text.append(
          mOutOfBagEvaluationObject.toSummaryString(
              "\n\n*** Out-of-bag estimates ***\n", getOutputOutOfBagComplexityStatistics()));
    }

    return text.toString();
  }

  /** Builds the classifier to generate a partition. */
  @Override
  public void generatePartition(Instances data) throws Exception {

    if (m_Classifier instanceof PartitionGenerator) buildClassifier(data);
    else throw ModelingException.ExceptionType.BAGGING_PARALLEL_ERROR.get(getPartitionErrorMsg());
  }

  /** Devuelve el mensaje de error estándar para particiones no soportadas. */
  private String getPartitionErrorMsg() {
    return "Classifier: " + getClassifierSpec() + " cannot generate a partition";
  }

  /** Computes an array that indicates leaf membership */
  @Override
  public double[] getMembershipValues(Instance inst) throws Exception {

    if (m_Classifier instanceof PartitionGenerator) {
      ArrayList<double[]> al = new ArrayList<>();
      int size = 0;
      for (Classifier mClassifier : m_Classifiers) {
        double[] r = ((PartitionGenerator) mClassifier).getMembershipValues(inst);
        size += r.length;
        al.add(r);
      }
      double[] values = new double[size];
      int pos = 0;
      for (double[] v : al) {
        System.arraycopy(v, 0, values, pos, v.length);
        pos += v.length;
      }
      return values;
    } else throw ModelingException.ExceptionType.BAGGING_PARALLEL_ERROR.get(getPartitionErrorMsg());
  }

  /** Returns the number of elements in the partition. */
  @Override
  public int numElements() throws ModelingException {

    if (m_Classifier instanceof PartitionGenerator) {
      try {
        int size = 0;
        for (Classifier mClassifier : m_Classifiers) {
          size += ((PartitionGenerator) mClassifier).numElements();
        }
        return size;
      } catch (Exception e) {
        throw ModelingException.ExceptionType.BAGGING_PARALLEL_ERROR.get(getPartitionErrorMsg(), e);
      }
    } else {
      throw ModelingException.ExceptionType.BAGGING_PARALLEL_ERROR.get(getPartitionErrorMsg());
    }
  }

  /**
   * Returns the revision string.
   *
   * @return the revision
   */
  @Override
  public String getRevision() {
    return RevisionUtils.extract("$Revision$");
  }

  /**
   * Main method for testing this class.
   *
   * @param argv the options
   */
  public static void main(String[] argv) {
    runClassifier(new Bagging(), argv);
  }

  protected List<Classifier> mClassifiersCache;

  /**
   * Aggregate an object with this one
   *
   * @param toAggregate the object to aggregate
   * @return the result of aggregation
   */
  @Override
  public Bagging aggregate(Bagging toAggregate) {
    if (!m_Classifier.getClass().isAssignableFrom(toAggregate.m_Classifier.getClass())) {
      throw ModelingException.ExceptionType.BAGGING_PARALLEL_ERROR.get(
          "Can't aggregate because base classifiers differ");
    }

    if (mClassifiersCache == null) {
      mClassifiersCache = new ArrayList<>();
      mClassifiersCache.addAll(Arrays.asList(m_Classifiers));
    }
    mClassifiersCache.addAll(Arrays.asList(toAggregate.m_Classifiers));

    return this;
  }

  /**
   * Call to complete the aggregation process. Allows implementers to do any final processing based
   * on how many objects were aggregated.
   */
  @Override
  public void finalizeAggregation() {
    m_Classifiers = mClassifiersCache.toArray(new Classifier[1]);
    m_NumIterations = m_Classifiers.length;

    mClassifiersCache = null;
  }

  @Override
  public TechnicalInformation getTechnicalInformation() {
    return null;
  }
}
