package tomocomd.searchmodels.v3;

import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;
import tomocomd.classifiers.ClassifierNameEnum;
import tomocomd.restart.SearchTask;
import tomocomd.searchmodels.v3.utils.MetricType;
import tomocomd.searchmodels.v3.utils.SearchPath;
import tomocomd.utils.ModelingException;
import tomocomd.utils.ReadData;
import weka.attributeSelection.ASSearch;
import weka.core.Instances;

public class InitSearchModel implements Serializable {

  private static final long serialVersionUID = 1L;

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

  // for save the status
  private final List<SearchTask> tasks;

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

    tasks = new ArrayList<>();
  }

  public void initSearchModel() {

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
                  new LinkedList<>(externalTestNames),
                  copyExternalData(externalTestData),
                  pathToSave,
                  trainData.classIndex(),
                  metricType,
                  new LinkedList<>(classifierNameSubList));

          tasks.add(
              new SearchTask(searchModelEvaluator, makeCopy(search), new Instances(trainData)));
        }
      }
    }

    submitStartSearch();
  }

  public void submitStartSearch() {
    ExecutorService executorService = Executors.newWorkStealingPool();
    List<Future<?>> futures = new ArrayList<>();
    try {
      for (SearchTask task : tasks) {
        if (!task.isCompleted()) {
          Future<?> future = executorService.submit(task);
          futures.add(future);
        }
      }

      for (Future<?> future : futures) {
        try {
          future.get();
        } catch (Exception e) {
          e.printStackTrace();
          System.exit(-1);
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

  private ASSearch makeCopy(ASSearch search) {
    try {
      return ASSearch.makeCopies(search, 1)[0];
    } catch (Exception e) {
      throw new RuntimeException("Error making copy of search algorithm", e);
    }
  }
}
