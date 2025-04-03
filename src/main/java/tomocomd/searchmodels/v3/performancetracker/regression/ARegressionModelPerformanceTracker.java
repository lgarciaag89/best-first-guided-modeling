package tomocomd.searchmodels.v3.performancetracker.regression;

import java.util.Objects;
import java.util.Random;
import java.util.Set;
import tomocomd.ModelingException;
import tomocomd.searchmodels.v3.performancetracker.AModelPerformanceTracker;
import tomocomd.searchmodels.v3.performancetracker.metricvalues.IMetricValue;
import tomocomd.searchmodels.v3.performancetracker.metricvalues.RegressionMetricValues;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;

public abstract class ARegressionModelPerformanceTracker extends AModelPerformanceTracker {

  protected ARegressionModelPerformanceTracker() {}

  @Override
  public boolean isClassification() {
    return false;
  }

  @Override
  protected void evaluateTrainData(Classifier classifier, Instances train)
      throws ModelingException {
    try {
      Evaluation rfEvaTrain = new Evaluation(train);
      rfEvaTrain.crossValidateModel(classifier, train, 10, new Random(1));
      double rST = rfEvaTrain.correlationCoefficient();
      double maeT = rfEvaTrain.meanAbsoluteError();
      double rmseT = rfEvaTrain.rootMeanSquaredError();

      trainValues = new RegressionMetricValues(rST * rST, maeT, rmseT);
    } catch (Exception e) {
      throw ModelingException.ExceptionType.BUILDING_MODEL_EXCEPTION.get(
          "Problems evaluating " + clasName + " model in training subset", e);
    }
  }

  @Override
  protected IMetricValue evaluateTestData(Classifier classifier, Instances train, Instances tune)
      throws ModelingException {
    try {
      Evaluation rfEvaTest = new Evaluation(train);
      rfEvaTest.evaluateModel(classifier, tune);
      double rSE = rfEvaTest.correlationCoefficient();
      double maeE = rfEvaTest.meanAbsoluteError();
      double rmseE = rfEvaTest.rootMeanSquaredError();

      return new RegressionMetricValues(rSE * rSE, maeE, rmseE);

    } catch (Exception e) {
      throw ModelingException.ExceptionType.BUILDING_MODEL_EXCEPTION.get(
          "Problems evaluating " + clasName + " model in " + tune.relationName() + " test subset",
          e);
    }
  }

  @Override
  public String getLineFromModelPerformance(long id, Set<String> mdNames) {
    RegressionMetricValues train = (RegressionMetricValues) this.trainValues;
    String res =
        String.format(
            "%s,%d,%d,%.5f,%.5f,%.5f",
            clasName, id, mdNames.size(), train.getQ2(), train.getMae(), train.getRmse());

    if (Objects.nonNull(testValues)) {
      RegressionMetricValues test = (RegressionMetricValues) this.testValues;
      res = String.format("%s,%.5f,%.5f,%.5f", res, test.getQ2(), test.getMae(), test.getRmse());
    }

    if (!externalTestValues.isEmpty()) {
      for (IMetricValue iMetricValue : externalTestValues) {
        RegressionMetricValues cMetricValue = (RegressionMetricValues) iMetricValue;
        res =
            String.format(
                "%s,%.5f,%.5f,%.5f",
                res, cMetricValue.getQ2(), cMetricValue.getMae(), cMetricValue.getRmse());
      }
    }
    return res + "," + mdNames.toString().replace(",", " ");
  }
}
