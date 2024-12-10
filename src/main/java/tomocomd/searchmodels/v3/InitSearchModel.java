package tomocomd.searchmodels.v3;

import java.io.File;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.Collectors;
import tomocomd.ModelingException;
import tomocomd.searchmodels.v3.utils.MetricType;
import tomocomd.searchmodels.v3.utils.SearchPath;
import tomocomd.utils.ReadData;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.AttributeSelection;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

public class InitSearchModel {

  private final File csvTrainFileName;
  private final Instances trainData;

  private final Instances testData;
  private final String csvTuneFileName;

  private final List<String> externalTestNames;
  private final List<Instances> externalTestData;

  private final SearchPath searchPath;
  private final List<MetricType> metricTypes;
  private final List<ASSearch> searchAlgorithms;
  private final List<AbstractClassifier> classifiers;

  public InitSearchModel(
      File csvFile,
      File tuneCsv,
      File folderExtCsvs,
      String act,
      List<AbstractClassifier> classifierList,
      List<ASSearch> searchList,
      List<MetricType> metricTypes,
      SearchPath searchPath,
      boolean isClassification)
      throws ModelingException {

    this.searchPath = searchPath;
    this.metricTypes = new LinkedList<>(metricTypes);
    this.searchAlgorithms = new LinkedList<>(searchList);
    this.classifiers = new LinkedList<>(classifierList);
    this.csvTrainFileName = csvFile;

    trainData = ReadData.readTrainData(csvFile, act, isClassification);
    // get tune data
    testData = ReadData.readTuneData(tuneCsv, trainData.classIndex(), isClassification);
    this.csvTuneFileName = Objects.nonNull(testData) ? testData.relationName() : null;

    externalTestData =
        ReadData.loadingExternalTestPath(folderExtCsvs, trainData.classIndex(), isClassification);
    externalTestNames =
        externalTestData.stream().map(Instances::relationName).collect(Collectors.toList());
  }

  public void initSearchModel() {

    ExecutorService executorService = Executors.newWorkStealingPool();
    List<Runnable> tasks = new ArrayList<>();

    List<List<AbstractClassifier>> listClassifiers =
        searchPath.equals(SearchPath.SHORT)
            ? Collections.singletonList(classifiers)
            : classifiers.stream().map(Collections::singletonList).collect(Collectors.toList());

    // generating the list of tasks
    for (ASSearch search : searchAlgorithms) {
      for (MetricType metricType : metricTypes) {
        for (List<AbstractClassifier> classifierSubList : listClassifiers) {
          String pathToSave =
              String.format(
                  "%s_models%s_%s_%s.csv",
                  csvTrainFileName.getAbsolutePath(),
                  classifierSubList.size() == 1
                      ? "_" + classifierSubList.get(0).getClass().getSimpleName()
                      : "",
                  metricType.toString(),
                  search.getClass().getSimpleName());
          SearchModelEvaluator searchModelEvaluator =
              new SearchModelEvaluator(
                  csvTrainFileName.getName(),
                  new Instances(trainData),
                  csvTuneFileName,
                  new Instances(testData),
                  externalTestNames,
                  copyExternalData(externalTestData),
                  pathToSave,
                  trainData.classIndex(),
                  metricType,
                  classifierSubList);

          tasks.add(() -> startSearch(searchModelEvaluator, search));
        }
      }
    }

    List<Future<?>> futures = new ArrayList<>();
    try {
      for (Runnable task : tasks) {
        Future<?> future = executorService.submit(task);
        futures.add(future);
      }

      for (Future<?> future : futures) {
        try {
          future.get();
        } catch (Exception e) {
          e.printStackTrace();
        }
      }
    } finally {
      executorService.shutdown();
    }
  }

  private List<Instances> copyExternalData(List<Instances> externalTestData) {
    if (Objects.isNull(externalTestData)) return Collections.emptyList();
    return externalTestData.stream()
        .map(
            instances -> {
              Instances copy = new Instances(instances);
              copy.setRelationName(instances.relationName());
              return copy;
            })
        .collect(Collectors.toList());
  }

  private void startSearch(SearchModelEvaluator searchModelEvaluator, ASSearch search) {
    AttributeSelection asSubset = new AttributeSelection();
    asSubset.setEvaluator(searchModelEvaluator);
    asSubset.setSearch(search);
    asSubset.setXval(false);

    try {
      Instances startTrain = new Instances(trainData);
      asSubset.SelectAttributes(startTrain);
      asSubset.selectedAttributes();
    } catch (Exception ex) {
      throw ModelingException.ExceptionType.BUILDING_MODEL_EXCEPTION.get(
          "Problems building classification models", ex);
    }
  }
}
