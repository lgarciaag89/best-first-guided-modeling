package tomocomd.searchmodels;

import weka.attributeSelection.ASSearch;
import weka.classifiers.AbstractClassifier;

public class SearchCombination {
  private final AbstractClassifier classifier;
  private final ASSearch aSSearch;
  private final RegressionOptimizationParam opt;

  public SearchCombination(
      AbstractClassifier classifier, ASSearch aSSearch, RegressionOptimizationParam opt) {
    this.classifier = classifier;
    this.aSSearch = aSSearch;
    this.opt = opt;
  }

  public AbstractClassifier getClassifier() {
    return classifier;
  }

  public ASSearch getaSSearch() {
    return aSSearch;
  }

  public RegressionOptimizationParam getOpt() {
    return opt;
  }
}
