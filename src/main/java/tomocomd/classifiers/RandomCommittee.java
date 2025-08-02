package tomocomd.classifiers;

/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    RandomCommittee.java
 *    Copyright (C) 2003-2012 University of Waikato, Hamilton, New Zealand
 *
 */

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import tomocomd.utils.ModelingException;
import weka.classifiers.AbstractClassifier;
import weka.core.*;

/*
 * Class based on Weka's RandomCommittee class.
 */
public class RandomCommittee extends RandomizableParallelEnhancer
    implements WeightedInstancesHandler, PartitionGenerator {

  private static final Logger logger = LogManager.getLogger(RandomCommittee.class);

  /** for serialization */
  private static final long serialVersionUID = 7003920439436055093L;

  /** training data */
  protected Instances mData;

  /** Constructor. */
  public RandomCommittee() {

    m_Classifier = new weka.classifiers.trees.RandomTree();
  }

  public RandomCommittee(int mSeed, int mNumExecutionSlots, String name) {
    super(mSeed, mNumExecutionSlots, name);
    m_Classifier = new weka.classifiers.trees.RandomTree();
    setBatchSize("100");
    setDebug(false);
    setDoNotCheckCapabilities(false);
    setNumDecimalPlaces(2);
    setNumIterations(100);
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
   * Builds the committee of randomizable classifiers.
   *
   * @param data the training data to be used for generating the bagged classifier.
   * @throws Exception if the classifier could not be built successfully
   */
  @Override
  public void buildClassifier(Instances data) throws Exception {

    // can classifier handle the data?
    getCapabilities().testWithFail(data);

    // get fresh instances
    mData = new Instances(data);
    super.buildClassifier(mData);

    if (!(m_Classifier instanceof Randomizable)) {
      throw new IllegalArgumentException("Base learner must implement Randomizable!");
    }

    m_Classifiers = AbstractClassifier.makeCopies(m_Classifier, m_NumIterations);

    Random random = mData.getRandomNumberGenerator(mSeed);

    // Resample data based on weights if base learner can't handle weights
    if (!(m_Classifier instanceof WeightedInstancesHandler)
        && !mData.allInstanceWeightsIdentical()) {
      mData = mData.resampleWithWeights(random);
    }

    for (weka.classifiers.Classifier mClassifier : m_Classifiers) {

      // Set the random number seed for the current classifier.
      ((Randomizable) mClassifier).setSeed(random.nextInt());
    }

    buildClassifiers();

    // save memory
    mData = null;
  }

  protected synchronized Instances getTrainingSet(int iteration) {
    // we don't manipulate the training data in any way.
    return mData;
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
      if (instance.classAttribute().isNumeric()) {
        double pred = m_Classifiers[i].classifyInstance(instance);
        if (!Utils.isMissingValue(pred)) {
          sums[0] += pred;
          numPreds++;
        }
      } else {
        newProbs = m_Classifiers[i].distributionForInstance(instance);
        for (int j = 0; j < newProbs.length; j++) sums[j] += newProbs[j];
      }
    }
    if (instance.classAttribute().isNumeric()) {
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

  @Override
  public void setBatchSize(String size) {

    if (getClassifier() instanceof BatchPredictor) {
      ((BatchPredictor) getClassifier()).setBatchSize(size);
    } else {
      super.setBatchSize(size);
    }
  }

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
   * BatchPredictor. Otherwise it simply calls the distributionForInstance() method repeatedly.
   *
   * @param insts the instances to get predictions for
   * @return an array of probability distributions, one for each instance
   * @throws Exception if a problem occurs
   */
  @Override
  public double[][] distributionsForInstances(Instances insts) throws Exception {

    if (getClassifier() instanceof BatchPredictor) {

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
        Future<double[][]> futureT =
            pool.submit(() -> calDdistributionsForInstances(insts, lo, hi));
        results.add(futureT);
      }

      // Form ensemble prediction
      double[][] ensemblePreds = ensemblePredictions(insts, results, pool);

      // Normalise ensemble predictions
      return normalizEnsemblePredictions(insts, ensemblePreds);
    }
    return distributionForInstancesForNoBatchPredictor(insts);
  }

  private double[][] ensemblePredictions(
      Instances insts, Set<Future<double[][]>> results, ExecutorService pool) {
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
    } finally {
      pool.shutdown();
    }
    return ensemblePreds;
  }

  private double[][] distributionForInstancesForNoBatchPredictor(Instances insts) throws Exception {
    double[][] result = new double[insts.numInstances()][insts.numClasses()];
    for (int i = 0; i < insts.numInstances(); i++) {
      result[i] = distributionForInstance(insts.instance(i));
    }
    return result;
  }

  private double[][] normalizEnsemblePredictions(Instances insts, double[][] ensemblePreds) {
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
    }

    for (double[] ensemblePred : ensemblePreds) {
      double sum = Utils.sum(ensemblePred);
      if (!Utils.eq((sum), 0)) {
        Utils.normalize(ensemblePred, sum);
      }
    }
    return ensemblePreds;
  }

  private double[][] calDdistributionsForInstances(Instances insts, int lo, int hi)
      throws Exception {
    if (insts.classAttribute().isNumeric()) {
      double[][] ensemblePreds = new double[insts.numInstances()][2];
      for (int i = lo; i < hi; i++) {
        double[][] preds = ((BatchPredictor) m_Classifiers[i]).distributionsForInstances(insts);
        for (int j1 = 0; j1 < preds.length; j1++) {
          if (!Utils.isMissingValue(preds[j1][0])) {
            ensemblePreds[j1][0] += preds[j1][0];
            ensemblePreds[j1][1]++;
          }
        }
      }
      return ensemblePreds;
    }

    double[][] ensemblePreds = new double[insts.numInstances()][insts.numClasses()];
    for (int i = lo; i < hi; i++) {
      double[][] preds = ((BatchPredictor) m_Classifiers[i]).distributionsForInstances(insts);
      for (int j1 = 0; j1 < preds.length; j1++) {
        for (int k = 0; k < preds[j1].length; k++) {
          ensemblePreds[j1][k] += preds[j1][k];
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

  /** Builds the classifier to generate a partition. */
  public void generatePartition(Instances data) throws Exception {

    if (m_Classifier instanceof PartitionGenerator) buildClassifier(data);
    else throw ModelingException.ExceptionType.RANDOMCOMMITEE_EXCEPTION.get(mssError());
  }

  /** Computes an array that indicates leaf membership */
  public double[] getMembershipValues(Instance inst) throws Exception {

    if (m_Classifier instanceof PartitionGenerator) {
      ArrayList<double[]> al = new ArrayList<>();
      int size = 0;
      for (weka.classifiers.Classifier mClassifier : m_Classifiers) {
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
    } else throw ModelingException.ExceptionType.RANDOMCOMMITEE_EXCEPTION.get(mssError());
  }

  /** Returns the number of elements in the partition. */
  public int numElements() throws Exception {

    if (m_Classifier instanceof PartitionGenerator) {
      int size = 0;
      for (weka.classifiers.Classifier mClassifier : m_Classifiers) {
        size += ((PartitionGenerator) mClassifier).numElements();
      }
      return size;
    } else throw ModelingException.ExceptionType.RANDOMCOMMITEE_EXCEPTION.get(mssError());
  }

  private String mssError() {
    return "Classifier: " + getClassifierSpec() + " cannot generate a partition";
  }
}
