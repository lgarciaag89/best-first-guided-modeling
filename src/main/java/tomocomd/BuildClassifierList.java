package tomocomd;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.stream.Collectors;
import weka.classifiers.AbstractClassifier;

public class BuildClassifierList {

  protected BuildClassifierList() {}

  public static List<AbstractClassifier> getclassifierList(
      String[] models, boolean isClassification) {
    Optional<String> hasExists =
        Arrays.stream(models).filter(model -> model.equals("all")).findFirst();
    if (hasExists.isPresent()) {
      return isClassification ? getClassifierList() : getRegressionList();
    }

    return Arrays.stream(models)
        .map(model -> getClassifier(model, isClassification))
        .filter(Objects::nonNull)
        .collect(Collectors.toList());
  }

  public static AbstractClassifier getClassifier(String name, boolean isClassification) {
    switch (name.toLowerCase()) {
      case "knn":
        return BuildClassifier.getKnnCV();
      case "randomforest":
        return BuildClassifier.getRandomForest();
      case "adaboost":
        return isClassification ? BuildClassifier.getAdaBoostM1() : null;
      case "additiveregression":
        return isClassification ? null : BuildClassifier.getAdditiveRegression();
      case "bayesnet":
        return isClassification ? BuildClassifier.getBayesNet() : null;
      case "logitboost":
        return isClassification ? BuildClassifier.getLogitBoost() : null;
      case "randomcommittee":
        return BuildClassifier.getRandomCommittee();
      case "smo-polykernel":
        return isClassification
            ? BuildClassifier.getSMOPolyKernel()
            : BuildClassifier.getSMORegPolyKernel();
      case "smo-puk":
        return isClassification ? BuildClassifier.getSMOPuk() : BuildClassifier.getSMORegPuk();
      case "linearegression":
        return isClassification ? null : BuildClassifier.getRegression();
      case "gaussian":
        return isClassification ? null : BuildClassifier.getGaussianProcess();
      case "bagging-smo":
        return isClassification
            ? BuildClassifier.getBaggingSMOPuk()
            : BuildClassifier.getBaggingSMOregPuk();
      case "bagging-knn":
        return BuildClassifier.getBaggingKnn();
      default:
        return null;
    }
  }

  private static List<AbstractClassifier> getClassifierList() {
    return Arrays.asList(
        BuildClassifier.getKnnCV(),
        BuildClassifier.getRandomForest(),
        BuildClassifier.getAdaBoostM1(),
        BuildClassifier.getBayesNet(),
        BuildClassifier.getLogitBoost(),
        BuildClassifier.getRandomCommittee(),
        BuildClassifier.getSMOPolyKernel(),
        BuildClassifier.getSMOPuk(),
        BuildClassifier.getBaggingSMOPuk(),
        BuildClassifier.getBaggingKnn());
  }

  private static List<AbstractClassifier> getRegressionList() {
    return Arrays.asList(
            BuildClassifier.getKnnCV(),
        BuildClassifier.getRandomForest(),
            BuildClassifier.getAdditiveRegression(),
            BuildClassifier.getRandomCommittee(),
            BuildClassifier.getSMORegPolyKernel(),
            BuildClassifier.getSMORegPuk(),
            BuildClassifier.getRegression(),
        BuildClassifier.getGaussianProcess(),
        BuildClassifier.getBaggingSMOregPuk(),
        BuildClassifier.getBaggingKnn());
  }
}
