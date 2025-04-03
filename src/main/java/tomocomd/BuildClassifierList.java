package tomocomd;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import tomocomd.searchmodels.v3.utils.MetricType;
import weka.classifiers.AbstractClassifier;

public class BuildClassifierList {

  protected BuildClassifierList() {}

  public static List<ClassifierNameEnum> getClassifierNameList(
      String[] models, boolean isClassification) {

    List<ClassifierNameEnum> validClassifiers =
        Arrays.stream(ClassifierNameEnum.values())
            .filter(
                enumValue ->
                    (isClassification
                            && enumValue.getProblemType() == MetricType.ProblemType.CLASSIFICATION)
                        || (!isClassification
                            && enumValue.getProblemType() == MetricType.ProblemType.REGRESSION)
                        || enumValue.getProblemType()
                            == MetricType.ProblemType.REGRESSION_CLASSIFICATION)
            .collect(Collectors.toList());

    if (Arrays.asList(models).contains("all")) {
      return validClassifiers;
    }

    return Arrays.stream(models)
        .map(
            name -> {
              try {
                ClassifierNameEnum classifier = ClassifierNameEnum.fromString(name);
                return validClassifiers.contains(classifier) ? classifier : null;
              } catch (IllegalArgumentException e) {
                System.out.println("Model not valid: " + name);
                return null;
              }
            })
        .filter(Objects::nonNull)
        .collect(Collectors.toList());
  }

  public static AbstractClassifier getClassifier(
      ClassifierNameEnum name, boolean isClassification) {
    switch (name) {
      case KNN:
        return BuildClassifier.getKnnCV();
      case RANDOMFOREST:
        return BuildClassifier.getRandomForest();
      case ADABOOST:
        return isClassification ? BuildClassifier.getAdaBoostM1() : null;
      case ADDITIVEREGRESSION:
        return isClassification ? null : BuildClassifier.getAdditiveRegression();
      case BAYESNET:
        return isClassification ? BuildClassifier.getBayesNet() : null;
      case LOGITBOOST:
        return isClassification ? BuildClassifier.getLogitBoost() : null;
      case RANDOMCOMMITTEE:
        return BuildClassifier.getRandomCommittee();
      case SMO_POLYKERNEL:
        return isClassification
            ? BuildClassifier.getSMOPolyKernel()
            : BuildClassifier.getSMORegPolyKernel();
      case SMO_PUK:
        return isClassification ? BuildClassifier.getSMOPuk() : BuildClassifier.getSMORegPuk();
      case LINEAREGRESSION:
        return isClassification ? null : BuildClassifier.getRegression();
      case GAUSSIAN:
        return isClassification ? null : BuildClassifier.getGaussianProcess();
      case BAGGING_SMO:
        return isClassification
            ? BuildClassifier.getBaggingSMOPuk()
            : BuildClassifier.getBaggingSMOregPuk();
      case BAGGING_KNN:
        return BuildClassifier.getBaggingKnn();
      default:
        return null;
    }
  }
}
