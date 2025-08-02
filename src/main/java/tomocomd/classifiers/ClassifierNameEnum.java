package tomocomd.classifiers;

import tomocomd.searchmodels.v3.utils.MetricType;

public enum ClassifierNameEnum {
  KNN("knn", MetricType.ProblemType.REGRESSION_CLASSIFICATION),
  RANDOMFOREST("randomforest", MetricType.ProblemType.REGRESSION_CLASSIFICATION),
  ADABOOST("adaboost", MetricType.ProblemType.CLASSIFICATION),
  ADDITIVEREGRESSION("additiveregression", MetricType.ProblemType.REGRESSION),
  BAYESNET("bayesnet", MetricType.ProblemType.CLASSIFICATION),
  LOGITBOOST("logitboost", MetricType.ProblemType.CLASSIFICATION),
  RANDOMCOMMITTEE("randomcommittee", MetricType.ProblemType.REGRESSION_CLASSIFICATION),
  SMO_POLYKERNEL("smo-polykernel", MetricType.ProblemType.REGRESSION_CLASSIFICATION),
  SMO_PUK("smo-puk", MetricType.ProblemType.REGRESSION_CLASSIFICATION),
  LINEAREGRESSION("linearegression", MetricType.ProblemType.REGRESSION),
  BAGGING_SMO("bagging-smo", MetricType.ProblemType.REGRESSION_CLASSIFICATION),
  BAGGING_KNN("bagging-knn", MetricType.ProblemType.REGRESSION_CLASSIFICATION);

  private final String classifierName;
  private final MetricType.ProblemType problemType;

  ClassifierNameEnum(String classifierName, MetricType.ProblemType regressionClassification) {
    this.classifierName = classifierName;
    this.problemType = regressionClassification;
  }

  public MetricType.ProblemType getProblemType() {
    return problemType;
  }

  public static ClassifierNameEnum fromString(String name) {
    for (ClassifierNameEnum value : ClassifierNameEnum.values()) {
      if (value.classifierName.equalsIgnoreCase(name)) {
        return value;
      }
    }
    throw new IllegalArgumentException("No enum constant found for classifier name: " + name);
  }
}
