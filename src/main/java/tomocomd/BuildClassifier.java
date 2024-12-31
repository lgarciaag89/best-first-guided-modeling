/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tomocomd;

import weka.attributeSelection.ASSearch;
import weka.attributeSelection.BestFirst;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.net.estimate.BayesNetEstimator;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.search.local.K2;
import weka.classifiers.bayes.net.search.local.LocalScoreSearchAlgorithm;
import weka.classifiers.bayes.net.search.local.Scoreable;
import weka.classifiers.functions.*;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.*;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.EuclideanDistance;
import weka.core.SelectedTag;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;

/** @author potter */
public class BuildClassifier {
  private BuildClassifier() {
    throw new IllegalStateException("BuildClassifier class");
  }

  public static AbstractClassifier getRandomForest() {
    RandomForest rf = new RandomForest();
    rf.setBagSizePercent(100);
    rf.setBatchSize("100");
    rf.setBreakTiesRandomly(false);
    rf.setCalcOutOfBag(false);
    rf.setDebug(false);
    rf.setDoNotCheckCapabilities(false);
    rf.setMaxDepth(0);
    rf.setNumDecimalPlaces(2);
    rf.setNumExecutionSlots(1);
    rf.setNumFeatures(0);
    rf.setNumIterations(100);
    rf.setOutputOutOfBagComplexityStatistics(false);
    rf.setPrintClassifiers(false);
    rf.setSeed(1);
    rf.setStoreOutOfBagPredictions(false);
    return rf;
  }

  public static AbstractClassifier getSMO() {
    SMO smo = new SMO();
    smo.setBatchSize("100");
    smo.setBuildCalibrationModels(false);
    smo.setC(1);
    smo.setChecksTurnedOff(false);
    smo.setDebug(false);
    smo.setDoNotCheckCapabilities(false);
    smo.setEpsilon(0.000000000001);
    smo.setNumDecimalPlaces(2);
    smo.setRandomSeed(1);
    smo.setNumFolds(-1);
    smo.setToleranceParameter(0.001);
    smo.setCalibrator(getLogistic());
    smo.setFilterType(new SelectedTag(SMO.FILTER_NORMALIZE, SMO.TAGS_FILTER));
    PolyKernel ker = new PolyKernel();
    ker.setCacheSize(250007);
    ker.setDebug(false);
    ker.setExponent(1);
    ker.setUseLowerOrder(false);
    smo.setKernel(ker);
    return smo;
  }

  public static AbstractClassifier getRegression() {
    LinearRegression classifier = new LinearRegression();
    classifier.setAttributeSelectionMethod(
        new SelectedTag(LinearRegression.SELECTION_NONE, LinearRegression.TAGS_SELECTION));
    classifier.setEliminateColinearAttributes(false);
    return classifier;
  }

  public static AbstractClassifier getSMOReg() {
    SMOreg smo = new SMOreg();
    smo.setBatchSize("100");
    smo.setC(1);
    smo.setDebug(false);
    smo.setDoNotCheckCapabilities(false);
    smo.setNumDecimalPlaces(2);
    smo.setFilterType(new SelectedTag(SMO.FILTER_NORMALIZE, SMO.TAGS_FILTER));
    PolyKernel ker = new PolyKernel();
    ker.setCacheSize(250007);
    ker.setDebug(false);
    ker.setExponent(1);
    ker.setUseLowerOrder(false);
    smo.setKernel(ker);
    return smo;
  }

  public static AbstractClassifier getLogistic() {
    Logistic logistic = new Logistic();
    logistic.setBatchSize("100");
    logistic.setDebug(false);
    logistic.setDoNotCheckCapabilities(false);
    logistic.setNumDecimalPlaces(4);
    logistic.setMaxIts(-1);
    logistic.setRidge(0.00000001);
    logistic.setUseConjugateGradientDescent(false);
    return logistic;
  }

  public static AbstractClassifier getRandomTree() {
    RandomTree tree = new RandomTree();
    tree.setKValue(0);
    tree.setAllowUnclassifiedInstances(false);
    tree.setBatchSize("100");
    tree.setBreakTiesRandomly(false);
    tree.setDebug(false);
    tree.setDoNotCheckCapabilities(false);
    tree.setMaxDepth(0);
    tree.setMinNum(1);
    tree.setMinVarianceProp(0.001);
    tree.setNumDecimalPlaces(2);
    tree.setNumFolds(0);
    tree.setSeed(1);
    return tree;
  }

  public static AbstractClassifier getKnn(int k) {
    IBk ibk = new IBk(k);
    ibk.setBatchSize("100");
    ibk.setDebug(false);
    ibk.setDoNotCheckCapabilities(false);

    ibk.setCrossValidate(false);
    ibk.setDistanceWeighting(new SelectedTag(IBk.WEIGHT_NONE, IBk.TAGS_WEIGHTING));
    ibk.setMeanSquared(false);
    ibk.setNumDecimalPlaces(2);
    ibk.setWindowSize(0);

    NearestNeighbourSearch nns = new LinearNNSearch();
    nns.setMeasurePerformance(false);
    EuclideanDistance ed = new EuclideanDistance();
    ed.setDontNormalize(false);
    ed.setInvertSelection(false);
    ed.setAttributeIndices("first-last");
    try {
      nns.setDistanceFunction(ed);
    } catch (Exception e) {
      e.printStackTrace();
    }
    ibk.setNearestNeighbourSearchAlgorithm(nns);
    return ibk;
  }

  public static AbstractClassifier getKnnCV(int k) {
    IBk ibk = new IBk(k);
    ibk.setBatchSize("100");
    ibk.setDebug(false);
    ibk.setDoNotCheckCapabilities(false);

    ibk.setCrossValidate(true);
    ibk.setDistanceWeighting(new SelectedTag(IBk.WEIGHT_NONE, IBk.TAGS_WEIGHTING));
    ibk.setMeanSquared(false);
    ibk.setNumDecimalPlaces(2);
    ibk.setWindowSize(0);

    NearestNeighbourSearch nns = new LinearNNSearch();
    nns.setMeasurePerformance(false);
    EuclideanDistance ed = new EuclideanDistance();
    ed.setDontNormalize(false);
    ed.setInvertSelection(false);
    ed.setAttributeIndices("first-last");
    try {
      nns.setDistanceFunction(ed);
    } catch (Exception e) {
      e.printStackTrace();
    }
    ibk.setNearestNeighbourSearchAlgorithm(nns);
    return ibk;
  }

  public static AbstractClassifier getNaiveBayes() {
    NaiveBayes nb = new NaiveBayes();
    nb.setBatchSize("100");
    nb.setDebug(false);
    nb.setDoNotCheckCapabilities(false);
    nb.setNumDecimalPlaces(2);

    nb.setDisplayModelInOldFormat(false);
    nb.setUseKernelEstimator(false);
    nb.setUseSupervisedDiscretization(false);
    return nb;
  }

  public static AbstractClassifier getJ48() {
    J48 j48 = new J48();
    j48.setBatchSize("100");
    j48.setDebug(false);
    j48.setDoNotCheckCapabilities(false);
    j48.setNumDecimalPlaces(2);

    j48.setBinarySplits(false);
    j48.setCollapseTree(true);
    j48.setConfidenceFactor((float) 0.25);
    j48.setDoNotMakeSplitPointActualValue(false);
    j48.setMinNumObj(2);
    j48.setNumFolds(3);
    j48.setReducedErrorPruning(false);
    j48.setSaveInstanceData(false);
    j48.setSeed(1);
    j48.setSubtreeRaising(true);
    j48.setUnpruned(false);
    j48.setUseLaplace(false);
    j48.setUseMDLcorrection(true);
    return j48;
  }

  public static AbstractClassifier getsimpleLogistic() {
    SimpleLogistic sl = new SimpleLogistic();
    sl.setBatchSize("100");
    sl.setDebug(false);
    sl.setDoNotCheckCapabilities(false);
    sl.setNumDecimalPlaces(2);

    sl.setErrorOnProbabilities(false);
    sl.setHeuristicStop(50);
    sl.setMaxBoostingIterations(500);
    sl.setNumBoostingIterations(0);
    sl.setUseAIC(false);
    sl.setUseCrossValidation(true);
    sl.setWeightTrimBeta(0);
    return sl;
  }

  public static AbstractClassifier getBayesNet() {
    BayesNet by = new BayesNet();
    by.setBatchSize("100");
    by.setDebug(false);
    by.setDoNotCheckCapabilities(false);
    by.setNumDecimalPlaces(2);

    by.setUseADTree(false);

    BayesNetEstimator bne = new SimpleEstimator();
    bne.setAlpha(0.5);
    by.setEstimator(bne);

    K2 sa = new K2();
    sa.setInitAsNaiveBayes(true);
    sa.setMarkovBlanketClassifier(false);
    sa.setMaxNrOfParents(1);
    sa.setRandomOrder(false);
    sa.setScoreType(new SelectedTag(Scoreable.BAYES, LocalScoreSearchAlgorithm.TAGS_SCORE_TYPE));
    by.setSearchAlgorithm(sa);
    return by;
  }

  public static AbstractClassifier getAdaBoostM1() {
    AdaBoostM1 ab = new AdaBoostM1();
    ab.setBatchSize("100");
    ab.setDebug(false);
    ab.setDoNotCheckCapabilities(false);
    ab.setNumDecimalPlaces(2);

    ab.setNumIterations(10);
    ab.setSeed(1);
    ab.setUseResampling(false);
    ab.setWeightThreshold(100);
    DecisionStump ds = new DecisionStump();
    ab.setClassifier(ds);
    return ab;
  }

  public static AbstractClassifier getLogitBoost() {
    LogitBoost lb = new LogitBoost();
    lb.setBatchSize("100");
    lb.setDebug(false);
    lb.setDoNotCheckCapabilities(false);
    lb.setNumDecimalPlaces(2);

    lb.setNumIterations(10);
    lb.setSeed(1);
    lb.setUseResampling(false);
    lb.setWeightThreshold(100);
    DecisionStump ds = new DecisionStump();
    lb.setClassifier(ds);

    lb.setZMax(3);
    lb.setLikelihoodThreshold(-1.7976931348623157E308);
    lb.setNumThreads(1);
    lb.setPoolSize(1);
    lb.setShrinkage(1);
    lb.setUseEstimatedPriors(false);
    lb.setUseResampling(false);
    return lb;
  }

  public static AbstractClassifier getRandomCommittee() {
    RandomCommittee rc = new RandomCommittee();
    rc.setBatchSize("100");
    rc.setDebug(false);
    rc.setDoNotCheckCapabilities(false);
    rc.setNumDecimalPlaces(2);

    rc.setNumIterations(10);
    rc.setSeed(1);
    rc.setNumExecutionSlots(1);

    rc.setClassifier(getRandomForest());
    return rc;
  }

  public static AbstractClassifier getPLS() {
    return new PLSClassifier();
  }

  public static AbstractClassifier getRacedIncrementalLogitBoost() {
    return new RacedIncrementalLogitBoost();
  }

  public static AbstractClassifier getGradient() {
    return new Grading();
  }

  public static AbstractClassifier getMultiBoost() {
    return new MultiBoostAB();
  }

  public static AbstractClassifier getSVM() {
    return new LibSVM();
  }

  public static ASSearch getBestFirst() throws Exception {
    BestFirst bf = new BestFirst();
    bf.setOptions(new String[] {"-D", "2"});
    return bf;
  }
}
