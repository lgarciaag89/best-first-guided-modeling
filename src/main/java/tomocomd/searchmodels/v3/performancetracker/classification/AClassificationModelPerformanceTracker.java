package tomocomd.searchmodels.v3.performancetracker.classification;

import java.util.Objects;
import java.util.Random;
import java.util.Set;
import tomocomd.ModelingException;
import tomocomd.searchmodels.v3.performancetracker.AModelPerformanceTracker;
import tomocomd.searchmodels.v3.performancetracker.metricvalues.ClassificationMetricValues;
import tomocomd.searchmodels.v3.performancetracker.metricvalues.IMetricValue;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;

public abstract class AClassificationModelPerformanceTracker extends AModelPerformanceTracker {

  protected AClassificationModelPerformanceTracker() {}

  @Override
  public boolean isClassification() {
    return true;
  }

  @Override
  protected void evaluateTrainData(Classifier classifier, Instances train)
      throws ModelingException {
    try {
      Evaluation rfEvaTrain = new Evaluation(train);

      rfEvaTrain.crossValidateModel(classifier, train, 10, new Random(1));

      double trainACC = rfEvaTrain.pctCorrect();
      double trainSen = recall(rfEvaTrain.confusionMatrix()) * 100;
      double trainSpe = specificity(rfEvaTrain.confusionMatrix()) * 100;
      double trainMCC = rfEvaTrain.weightedMatthewsCorrelation();

      trainValues = new ClassificationMetricValues(trainACC, trainSen, trainSpe, trainMCC);
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
      double testACC = rfEvaTest.pctCorrect();
      double testSen = recall(rfEvaTest.confusionMatrix()) * 100;
      double testSpe = specificity(rfEvaTest.confusionMatrix()) * 100;
      double testMCC = rfEvaTest.weightedMatthewsCorrelation();

      return new ClassificationMetricValues(testACC, testSen, testSpe, testMCC);

    } catch (Exception e) {
      throw ModelingException.ExceptionType.BUILDING_MODEL_EXCEPTION.get(
          "Problems evaluating " + clasName + " model in " + tune.relationName() + " test subset",
          e);
    }
  }

  @Override
  public String getLineFromModelPerformance(long id, Set<String> mdNames) {
    ClassificationMetricValues train = (ClassificationMetricValues) this.trainValues;
    String res =
        String.format(
            "%s,%d,%d,%.5f,%.5f,%.5f,%.5f",
            clasName,
            id,
            mdNames.size(),
            train.getAcc(),
            train.getSen(),
            train.getSpe(),
            train.getMcc());

    if (Objects.nonNull(testValues)) {
      ClassificationMetricValues test = (ClassificationMetricValues) this.testValues;
      res =
          String.format(
              "%s,%.5f,%.5f,%.5f,%.5f",
              res, test.getAcc(), test.getSen(), test.getSpe(), test.getMcc());
    }

    if (!externalTestValues.isEmpty()) {
      for (IMetricValue iMetricValue : externalTestValues) {
        ClassificationMetricValues cMetricValue = (ClassificationMetricValues) iMetricValue;
        res =
            String.format(
                "%s,%.5f,%.5f,%.5f,%.5f",
                res,
                cMetricValue.getAcc(),
                cMetricValue.getSen(),
                cMetricValue.getSpe(),
                cMetricValue.getMcc());
      }
    }
    return res + "," + mdNames.toString().replace(",", " ");
  }

  private double specificity(double[][] confusionMatrix) {
    double tn = confusionMatrix[0][0];
    double fp = confusionMatrix[0][1];
    return tn / (tn + fp);
  }

  private double recall(double[][] confusionMatrix) {
    double tp = confusionMatrix[1][1];
    double fn = confusionMatrix[1][0];
    return tp / (tp + fn);
  }
}
