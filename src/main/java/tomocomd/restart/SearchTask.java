package tomocomd.restart;

import java.io.Serializable;
import java.util.stream.Collectors;
import tomocomd.searchmodels.v3.SearchModelEvaluator;
import weka.attributeSelection.ASSearch;
import weka.core.Instances;

public class SearchTask implements Runnable, Serializable {

  private final SearchModelEvaluator evaluator;
  private final ASSearch search;
  private final Instances trainData;
  private boolean isCompleted;

  public SearchTask(SearchModelEvaluator evaluator, ASSearch search, Instances trainData) {
    this.evaluator = evaluator;
    this.search = search;
    this.trainData = new Instances(trainData);
    this.isCompleted = false;
  }

  @Override
  public void run() {
    String classifierName =
        evaluator.getClassifiersName().size() == 1
            ? evaluator.getClassifiersName().get(0).toString()
            : evaluator.getClassifiersName().stream()
                .map(Object::toString)
                .collect(Collectors.joining(","));
    System.out.printf(
        "Starting classifiers:[%s] with search: %s and metric: %s%n",
        classifierName, search.getClass().getSimpleName(), evaluator.getMetricType());

    weka.attributeSelection.AttributeSelection attributeSelections =
        new weka.attributeSelection.AttributeSelection();
    attributeSelections.setEvaluator(evaluator);
    attributeSelections.setSearch(search);
    attributeSelections.setXval(false);

    try {
      attributeSelections.SelectAttributes(trainData);
    } catch (Exception ex) {
      throw new RuntimeException("Problems building classification models", ex);
    }
    isCompleted = true;
    System.out.printf(
        "Completed classifiers:[%s] with search: %s and metric: %s%n",
        classifierName, search.getClass().getSimpleName(), evaluator.getMetricType());
  }

  public boolean isCompleted() {
    return isCompleted;
  }
}
