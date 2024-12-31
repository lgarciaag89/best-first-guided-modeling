package tomocomd.searchmodels.v3;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.apache.commons.cli.CommandLine;
import tomocomd.BuildClassifier;
import tomocomd.BuildClassifierList;
import tomocomd.searchmodels.v3.utils.MetricType;
import tomocomd.searchmodels.v3.utils.SearchPath;
import weka.attributeSelection.ASSearch;
import weka.classifiers.AbstractClassifier;

public class BuildModels {

  public static void buildModels(
      File trainFile, File tunePath, File extFolderPath, String act, CommandLine cmd) {

    boolean isClassification = cmd.hasOption('c');
    boolean isShort = cmd.hasOption('s');

    List<AbstractClassifier> classifierList =
        cmd.hasOption("m")
            ? BuildClassifierList.getclassifierList(cmd.getOptionValues("m"), isClassification)
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
              classifierList,
              asSearches,
              metrics,
              isShort ? SearchPath.SHORT : SearchPath.LONG,
              isClassification);
      initSearchModel.initSearchModel();
    } catch (Exception ex) {
      ex.printStackTrace();
      System.exit(-1);
    }
  }

  private static List<MetricType> getMetrics(boolean isClassification, boolean hasTune) {
    return isClassification
        ? hasTune
            ? Arrays.asList(MetricType.MCC_MEAN, MetricType.MCC_TRAIN, MetricType.MCC_TEST)
            : Collections.singletonList(MetricType.MCC_TRAIN)
        : hasTune
            ? Arrays.asList(MetricType.Q2_TRAIN, MetricType.Q2_EXT, MetricType.Q2_MEAN)
            : Collections.singletonList(MetricType.Q2_TRAIN);
  }
}
