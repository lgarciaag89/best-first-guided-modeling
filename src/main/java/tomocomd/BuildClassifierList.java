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
        return BuildClassifier.getKnnCV(10);
      case "randomforest":
        return BuildClassifier.getRandomForest();
      case "adaboost":
        return isClassification ? BuildClassifier.getAdaBoostM1() : null;
      case "bayesnet":
        return isClassification ? BuildClassifier.getBayesNet() : null;
      case "gradient":
        return isClassification ? BuildClassifier.getGradient() : null;
      case "logistic":
        return isClassification ? BuildClassifier.getLogistic() : null;
      case "logitboost":
        return isClassification ? BuildClassifier.getLogitBoost() : null;
      case "simplelogistic":
        return isClassification ? BuildClassifier.getsimpleLogistic() : null;
      case "multibost":
        return isClassification ? BuildClassifier.getMultiBoost() : null;
      case "naivebayes":
        return isClassification ? BuildClassifier.getNaiveBayes() : null;
      case "racedincrementallogitboost":
        return isClassification ? BuildClassifier.getRacedIncrementalLogitBoost() : null;
      case "randomcommittee":
        return BuildClassifier.getRandomCommittee();
      case "randomtree":
        return isClassification ? BuildClassifier.getRandomTree() : null;
      case "smo":
        return isClassification ? BuildClassifier.getSMO() : BuildClassifier.getSMOReg();
      case "svm":
        return isClassification ? BuildClassifier.getSVM() : null;
      case "linerregression":
        return isClassification ? null : BuildClassifier.getRegression();
      case "j48":
        return isClassification ? BuildClassifier.getJ48() : null;
      default:
        return null;
    }
  }

  private static List<AbstractClassifier> getClassifierList() {
    return Arrays.asList(
        BuildClassifier.getKnnCV(10),
        BuildClassifier.getRandomForest(),
        BuildClassifier.getAdaBoostM1(),
        BuildClassifier.getBayesNet(),
        BuildClassifier.getGradient(),
        BuildClassifier.getLogistic(),
        BuildClassifier.getLogitBoost(),
        BuildClassifier.getsimpleLogistic(),
        BuildClassifier.getMultiBoost(),
        BuildClassifier.getNaiveBayes(),
        BuildClassifier.getRacedIncrementalLogitBoost(),
        BuildClassifier.getRandomCommittee(),
        BuildClassifier.getRandomTree(),
        BuildClassifier.getSMO(),
        BuildClassifier.getSVM(),
        BuildClassifier.getRegression(),
        BuildClassifier.getJ48());
  }

  private static List<AbstractClassifier> getRegressionList() {
    return Arrays.asList(
        BuildClassifier.getRegression(),
        BuildClassifier.getKnnCV(10),
        BuildClassifier.getRandomForest(),
        BuildClassifier.getRandomCommittee(),
        BuildClassifier.getSMOReg());
  }
}
