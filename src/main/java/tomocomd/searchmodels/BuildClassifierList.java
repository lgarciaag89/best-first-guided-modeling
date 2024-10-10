package tomocomd.searchmodels;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import tomocomd.BuildClassifier;
import weka.classifiers.AbstractClassifier;

public class BuildClassifierList {

  protected BuildClassifierList() {}

  public static List<AbstractClassifier> getclassifierList(
      String[] models, boolean isClassification) {
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
        return BuildClassifier.getAdaBoostM1();
      case "bayesnet":
        return BuildClassifier.getBayesNet();
      case "gradient":
        return BuildClassifier.getGradient();
      case "logistic":
        return BuildClassifier.getLogistic();
      case "logitboost":
        return BuildClassifier.getLogitBoost();
      case "simplelogistic":
        return BuildClassifier.getsimpleLogistic();
      case "multibost":
        return BuildClassifier.getMultiBoost();
      case "naivebayes":
        return BuildClassifier.getNaiveBayes();
      case "racedincrementallogitboost":
        return BuildClassifier.getRacedIncrementalLogitBoost();
      case "randomcommittee":
        return BuildClassifier.getRandomCommittee();
      case "randomtree":
        return BuildClassifier.getRandomTree();
      case "smo":
        return isClassification ? BuildClassifier.getSMO() : BuildClassifier.getSMOReg();
      case "svm":
        return BuildClassifier.getSVM();
      case "linerregression":
        return BuildClassifier.getRegression();
      default:
        return null;
    }
  }
}
