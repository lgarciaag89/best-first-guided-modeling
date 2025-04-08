package tomocomd.searchmodels.v3;

import java.io.File;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.Collectors;
import tomocomd.ClassifierNameEnum;
import tomocomd.ModelingException;
import tomocomd.searchmodels.v3.utils.MetricType;
import tomocomd.searchmodels.v3.utils.SearchPath;
import tomocomd.utils.ReadData;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.AttributeSelection;
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
  private final List<ClassifierNameEnum> classifiersName;

  public InitSearchModel(
      File csvFile,
      File tuneCsv,
      File folderExtCsvs,
      String act,
      List<ClassifierNameEnum> classifierNameList,
      List<ASSearch> searchList,
      List<MetricType> metricTypes,
      SearchPath searchPath,
      boolean isClassification)
      throws ModelingException {

    this.searchPath = searchPath;
    this.metricTypes = new LinkedList<>(metricTypes);
    this.searchAlgorithms = new LinkedList<>(searchList);
    this.classifiersName = new LinkedList<>(classifierNameList);
    this.csvTrainFileName = csvFile;

    trainData = ReadData.readData(csvFile, act, isClassification);
    // get tune data
    testData = ReadData.readData(tuneCsv, act, isClassification);
    this.csvTuneFileName = Objects.nonNull(testData) ? testData.relationName() : null;

    externalTestData = ReadData.loadingExternalTestPath(folderExtCsvs, act, isClassification);
    externalTestNames =
        externalTestData.stream().map(Instances::relationName).collect(Collectors.toList());
  }

  public void initSearchModel() {

    ExecutorService executorService = Executors.newWorkStealingPool();
    List<Runnable> tasks = new ArrayList<>();

    List<List<ClassifierNameEnum>> listClassifiersName =
        searchPath.equals(SearchPath.SHORT)
            ? Collections.singletonList(classifiersName)
            : classifiersName.stream().map(Collections::singletonList).collect(Collectors.toList());

    // generating the list of tasks
    for (ASSearch search : searchAlgorithms) {
      for (MetricType metricType : metricTypes) {
        for (List<ClassifierNameEnum> classifierNameSubList : listClassifiersName) {
          String pathToSave =
              String.format(
                  "%s_models%s_%s_%s.csv",
                  csvTrainFileName.getAbsolutePath(),
                  classifierNameSubList.size() == 1
                      ? "_" + classifierNameSubList.get(0).toString()
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
                  classifierNameSubList);

          tasks.add(
              () -> {
                String classifierName =
                    classifierNameSubList.size() == 1
                        ? classifierNameSubList.get(0).toString()
                        : classifierNameSubList.stream()
                            .map(Object::toString)
                            .collect(Collectors.joining(","));
                System.out.printf(
                    "Starting clasifiers:[%s] with search: %s and metric: %s%n",
                    classifierName, search.getClass().getSimpleName(), metricType);
                startSearch(searchModelEvaluator, search);
                System.out.printf(
                    "Completed clasifiers:[%s] with search: %s and metric: %s%n",
                    classifierName, search.getClass().getSimpleName(), metricType);
              });
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
