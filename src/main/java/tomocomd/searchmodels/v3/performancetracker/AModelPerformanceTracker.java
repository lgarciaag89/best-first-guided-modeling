package tomocomd.searchmodels.v3.performancetracker;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import tomocomd.ModelingException;
import tomocomd.searchmodels.v3.performancetracker.metricvalues.IMetricValue;
import tomocomd.searchmodels.v3.utils.MetricType;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

public abstract class AModelPerformanceTracker {
  protected String clasName;

  protected IMetricValue trainValues;
  protected IMetricValue testValues;
  protected List<IMetricValue> externalTestValues;

  protected AModelPerformanceTracker() {}

  public abstract MetricType getMetricType();

  public abstract String getLineFromModelPerformance(long id, Set<String> mds);

  public abstract double computeValueToCompare() throws ModelingException;

  protected abstract void evaluateTrainData(Classifier classifier, Instances train)
      throws ModelingException;

  protected abstract IMetricValue evaluateTestData(
      Classifier classifier, Instances train, Instances tune) throws ModelingException;

  protected void getClassifierName(AbstractClassifier clasTmp) {
    if (clasTmp instanceof IBk) {
      clasName = String.format("KNN(%s)", clasTmp.toString().split(" ")[3]);
    } else if (clasTmp instanceof SMO) {
      clasName = String.format("SMO(%s)", ((SMO) clasTmp).getKernel().getClass().getSimpleName());
    } else if (clasTmp instanceof SMOreg) {
      clasName =
          String.format("SMO(%s)", ((SMOreg) clasTmp).getKernel().getClass().getSimpleName());
    } else if (clasTmp instanceof Bagging) {
      clasName = clasTmp instanceof RandomForest ? "RandomForest" :
          String.format(
              "Bagging(%s)", ((Bagging) clasTmp).getClassifier().getClass().getSimpleName());
    } else {
      clasName = clasTmp.getClass().getSimpleName();
    }
  }

  private Classifier[] buildAndGetClassifiers(
      AbstractClassifier classifier, Instances train, int numCopies) {
    try {
      AbstractClassifier clasTmp = (AbstractClassifier) AbstractClassifier.makeCopy(classifier);
      clasTmp.buildClassifier(train);
      getClassifierName(clasTmp);
      return AbstractClassifier.makeCopies(clasTmp, numCopies);
    } catch (Exception e) {
      throw ModelingException.ExceptionType.BUILDING_MODEL_EXCEPTION.get(
          "Problems building and copy " + clasName + " model", e);
    }
  }

  private void buildModelPerformance(
      AbstractClassifier classifier, Instances train, Instances tune, List<Instances> externalTest)
      throws ModelingException {
    int numCopies =
        1
            + (Objects.isNull(tune) ? 0 : 1)
            + (Objects.isNull(externalTest) ? 0 : externalTest.size());
    Classifier[] copiedModels = buildAndGetClassifiers(classifier, train, numCopies);

    evaluateTrainData(copiedModels[0], train);
    int startEvaluateModels = 1;
    if (Objects.nonNull(tune)) {
      testValues = evaluateTestData(copiedModels[startEvaluateModels++], train, tune);
    }

    if (Objects.nonNull(externalTest)) {
      externalTestValues = new ArrayList<>();
      for (Instances test : externalTest) {
        externalTestValues.add(evaluateTestData(copiedModels[startEvaluateModels++], train, test));
      }
    }
  }

  public double getModelPerformance(
      AbstractClassifier classifier,
      Instances train,
      Instances test,
      List<Instances> externalTest) {
    buildModelPerformance(classifier, train, test, externalTest);
    return computeValueToCompare();
  }
}
