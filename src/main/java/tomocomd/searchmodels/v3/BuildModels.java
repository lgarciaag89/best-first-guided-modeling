package tomocomd.searchmodels.v3;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.apache.commons.cli.CommandLine;
import tomocomd.classifiers.BuildClassifier;
import tomocomd.classifiers.BuildClassifierList;
import tomocomd.classifiers.ClassifierNameEnum;
import tomocomd.restart.ModelAutoSaver;
import tomocomd.restart.StatusManager;
import tomocomd.searchmodels.v3.utils.MetricType;
import tomocomd.searchmodels.v3.utils.SearchPath;
import weka.attributeSelection.ASSearch;

public class BuildModels {

  protected BuildModels() {
    // Prevent instantiation
    throw new UnsupportedOperationException("This class cannot be instantiated");
  }

  public static void buildModels(
      File trainFile, File tunePath, File extFolderPath, String act, CommandLine cmd) {

    boolean isClassification = cmd.hasOption('c');
    boolean isShort = cmd.hasOption('s');

    List<ClassifierNameEnum> classifierNameList =
        cmd.hasOption("m")
            ? BuildClassifierList.getClassifierNameList(cmd.getOptionValues("m"), isClassification)
            : Collections.emptyList();

    List<MetricType> metrics = getMetrics(isClassification, cmd.hasOption("t"));

    try {
      ArrayList<ASSearch> asSearches =
          new ArrayList<>(Collections.singletonList(BuildClassifier.getBestFirst()));
      InitSearchModel initSearchModel =
          new InitSearchModel(
              trainFile,
              tunePath,
              extFolderPath,
              act,
              classifierNameList,
              asSearches,
              metrics,
              isShort ? SearchPath.SHORT : SearchPath.LONG,
              isClassification);

      File saveFile = new File(trainFile.getAbsolutePath() + ".status");
      ModelAutoSaver autoSaver = new ModelAutoSaver(initSearchModel, saveFile);
      autoSaver.startAutoSave();

      initSearchModel.initSearchModel();
      autoSaver.stopAutoSave();
    } catch (Exception ex) {
      ex.printStackTrace();
      System.exit(-1);
    }
  }

  public static void restartModelling(File statusFile) throws IOException, ClassNotFoundException {
    InitSearchModel initSearchModel = StatusManager.loadStatus(statusFile);

    ModelAutoSaver autoSaver = new ModelAutoSaver(initSearchModel, statusFile);
    autoSaver.startAutoSave();

    initSearchModel.submitStartSearch();
    autoSaver.stopAutoSave();
  }

  private static List<MetricType> getMetrics(boolean isClassification, boolean hasTune) {

    if (isClassification) {
      return hasTune
          ? Arrays.asList(MetricType.MCC_MEAN, MetricType.MCC_TRAIN, MetricType.MCC_TEST)
          : Collections.singletonList(MetricType.MCC_TRAIN);
    }

    return hasTune
        ? Arrays.asList(MetricType.Q2_TRAIN, MetricType.Q2_EXT, MetricType.Q2_MEAN)
        : Collections.singletonList(MetricType.Q2_TRAIN);
  }
}
