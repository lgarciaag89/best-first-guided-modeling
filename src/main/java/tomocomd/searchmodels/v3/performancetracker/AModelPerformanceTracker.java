package tomocomd.searchmodels.v3.performancetracker;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import tomocomd.BuildClassifierList;
import tomocomd.ClassifierNameEnum;
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

  public abstract boolean isClassification();

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
      clasName =
          clasTmp instanceof RandomForest
              ? "RandomForest"
              : String.format(
                  "Bagging(%s)", ((Bagging) clasTmp).getClassifier().getClass().getSimpleName());
    } else {
      clasName = clasTmp.getClass().getSimpleName();
    }
  }

  public Classifier[] buildAndGetClassifiers(
      ClassifierNameEnum classifierName, Instances train, int numCopies) {
    try {
      AbstractClassifier classifier =
          BuildClassifierList.getClassifier(classifierName, isClassification());

      if (classifier instanceof IBk) {
        ((IBk) classifier).setKNN((int) Math.sqrt(train.numInstances()));
      }

      if (classifier instanceof Bagging && ((Bagging) classifier).getClassifier() instanceof IBk) {
        ((IBk) ((Bagging) classifier).getClassifier())
            .setKNN((int) Math.sqrt(train.numInstances()));
      }

      classifier.buildClassifier(train);
      getClassifierName(classifier);
      if (classifierName == ClassifierNameEnum.GAUSSIAN) {
        Classifier[] classifiers = new Classifier[numCopies];
        classifiers[0] = classifier;
        for (int i = 1; i < numCopies; i++) {
          classifiers[i] = BuildClassifierList.getClassifier(classifierName, isClassification());
          classifiers[i].buildClassifier(train);
        }
        return classifiers;
      }

      return AbstractClassifier.makeCopies(classifier, numCopies);
    } catch (Exception e) {
      throw ModelingException.ExceptionType.BUILDING_MODEL_EXCEPTION.get(
          "Problems building and copy " + classifierName + " model", e);
    }
  }

  private void buildModelPerformance(
      ClassifierNameEnum classifierName,
      Instances train,
      Instances tune,
      List<Instances> externalTest)
      throws ModelingException {
    int numCopies =
        1
            + (Objects.isNull(tune) ? 0 : 1)
            + (Objects.isNull(externalTest) ? 0 : externalTest.size());
    Classifier[] copiedModels = buildAndGetClassifiers(classifierName, train, numCopies);

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
      ClassifierNameEnum classifierName,
      Instances train,
      Instances test,
      List<Instances> externalTest) {
    buildModelPerformance(classifierName, train, test, externalTest);
    return computeValueToCompare();
  }
}
