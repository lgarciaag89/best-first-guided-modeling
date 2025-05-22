package tomocomd.searchmodels.v3;

import java.util.*;
import java.util.stream.IntStream;
import tomocomd.ClassifierNameEnum;
import tomocomd.ModelingException;
import tomocomd.searchmodels.v3.performancetracker.AModelPerformanceTracker;
import tomocomd.searchmodels.v3.performancetracker.ModelPerformanceTrackerFactory;
import tomocomd.searchmodels.v3.utils.MetricType;
import tomocomd.searchmodels.v3.utils.printhead.APrintResult;
import tomocomd.searchmodels.v3.utils.printhead.PrintResultFactory;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.SubsetEvaluator;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class SearchModelEvaluator extends ASEvaluation implements SubsetEvaluator {

  protected final String trainPath;
  protected final Instances trainTotal;

  protected final Instances testTotal;
  protected final String testPath;

  protected final List<String> externalTestPath;
  protected final List<Instances> externalTestTotal;

  protected final String pathToSave;
  protected final int classAct;
  protected long modelId;
  protected final MetricType metricType;
  protected final List<ClassifierNameEnum> classifiersName;
  protected final APrintResult printResult;

  public List<ClassifierNameEnum> getClassifiersName() {
    return classifiersName;
  }

  public MetricType getMetricType() {
    return metricType;
  }

  public SearchModelEvaluator(
      String trainPath,
      Instances trainTotal,
      String testPath,
      Instances testTotal,
      List<String> externalTestPath,
      List<Instances> externalTestTotal,
      String pathToSave,
      int classAct,
      MetricType metricType,
      List<ClassifierNameEnum> classifiersName) {
    this.trainPath = trainPath;
    this.trainTotal = trainTotal;
    this.testTotal = testTotal;
    this.testPath = testPath;
    this.externalTestPath = externalTestPath;
    this.externalTestTotal = externalTestTotal;
    this.pathToSave = pathToSave;
    this.classAct = classAct;
    this.modelId = 0;
    this.metricType = metricType;
    this.classifiersName = classifiersName;
    this.printResult = PrintResultFactory.createPrintResult(metricType);
    printResult.createHead(Objects.nonNull(testPath), externalTestPath, pathToSave);
  }

  @Override
  public void buildEvaluator(Instances data) {}

  @Override
  public double evaluateSubset(BitSet bitset) throws ModelingException {

    int[] idx =
        IntStream.concat(
                Arrays.stream(new int[] {classAct}),
                IntStream.of(bitset.stream().sorted().toArray()))
            .toArray();
    if (idx.length == 1) return 0;

    Instances train;
    Instances test = null;
    List<Instances> externalTest = new LinkedList<>();

    if (idx.length < trainTotal.numAttributes()) {
      train = remove(trainTotal, idx);

      if (Objects.nonNull(testTotal)) test = remove(testTotal, idx);

      for (Instances instances : externalTestTotal) {
        externalTest.add(remove(instances, idx));
      }

    } else {
      train = new Instances(trainTotal);
      if (Objects.nonNull(testTotal)) test = new Instances(testTotal);

      externalTest.addAll(externalTestTotal);
    }

    return getEvaluation(train, test, externalTest);
  }

  protected double getEvaluation(
      Instances train, Instances internalTest, List<Instances> externalTests)
      throws ModelingException {

    Set<String> mdNames = getMDNames(train);

    return classifiersName.stream()
        .map(
            classifier -> {
              AModelPerformanceTracker tracker =
                  ModelPerformanceTrackerFactory.createModelPerformanceTracker(metricType);
              try {
                double valueToCompare =
                    tracker.getModelPerformance(classifier, train, internalTest, externalTests);

                String line = tracker.getLineFromModelPerformance(modelId++, mdNames);
                printResult.printLine(line, pathToSave);

                return valueToCompare;
              } catch (Exception e) {
                return Double.MIN_VALUE;
              }
            })
        .max(Double::compare)
        .orElse(Double.NEGATIVE_INFINITY);
  }

  protected Set<String> getMDNames(Instances data) {
    Set<String> mdNames = new LinkedHashSet<>();
    for (int i = 0; i < data.numAttributes(); i++) {
      if (i != data.classIndex()) {
        mdNames.add(data.attribute(i).name());
      }
    }
    return mdNames;
  }

  protected Instances remove(Instances data, int[] pos) throws ModelingException {
    Remove rem = new Remove();
    rem.setAttributeIndicesArray(pos);
    rem.setInvertSelection(true);
    try {
      rem.setInputFormat(data);
      return Filter.useFilter(data, rem);
    } catch (Exception ex) {
      throw ModelingException.ExceptionType.LOAD_DATASET_EXCEPTION.get(
          String.format("Problems loading dataset:%s", data.relationName()), ex);
    }
  }
}
