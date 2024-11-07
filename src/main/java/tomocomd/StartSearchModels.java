package tomocomd;

import java.io.File;
import java.util.*;
import org.apache.commons.cli.*;
import tomocomd.searchmodels.deprecated.BuildClassifierList;
import tomocomd.searchmodels.v3.InitSearchModel;
import tomocomd.searchmodels.v3.utils.DefiningCMDOptions;
import tomocomd.searchmodels.v3.utils.MetricType;
import tomocomd.searchmodels.v3.utils.SearchPath;
import weka.attributeSelection.ASSearch;
import weka.classifiers.AbstractClassifier;

public class StartSearchModels {
  public static void main(String[] args) {
    parseOptions(args);
  }

  private static void parseOptions(String[] args) {

    CommandLine cmd = DefiningCMDOptions.getCommandLine(args);

    // train file
    File trainFile = null;
    try {
      trainFile = new File(cmd.getOptionValue("t"));
    } catch (Exception ex) {
      System.err.println("Error loading train dataset:" + ex.getMessage());
      System.exit(-1);
    }
    System.out.println("Train file: " + trainFile.getAbsolutePath());

    // target
    String act = "";
    if (cmd.hasOption("e")) {
      act = cmd.getOptionValue("e");
    }
    if (act.isEmpty()) {
      System.err.println("It is necessary define endpoint class");
      System.exit(-1);
    }
    System.out.println("Endpoint: " + act);

    // tune file
    File tunePath = null;
    if (cmd.hasOption("p")) {
      tunePath = new File(cmd.getOptionValue("p"));
      System.out.println("Tune file: " + tunePath.getAbsolutePath());
    }

    // external folder
    File extFolderPath = null;
    if (cmd.hasOption("x")) {
      if (!new File(cmd.getOptionValue("x")).exists()) {
        System.err.println("External folder " + cmd.getOptionValue("x") + " does not exist");
        System.exit(-1);
      }
      extFolderPath = new File(cmd.getOptionValue("x"));
      System.out.println("External folder: " + extFolderPath.getAbsolutePath());
    }

    boolean isClassification = cmd.hasOption('c');
    boolean isShort = cmd.hasOption('s');

    List<AbstractClassifier> classifierList =
        cmd.hasOption("m")
            ? BuildClassifierList.getclassifierList(cmd.getOptionValues("m"), isClassification)
            : Collections.emptyList();

    ArrayList<ASSearch> asSearches =
        new ArrayList<>(Collections.singletonList(BuildClassifier.getBestFirst()));
    List<MetricType> metrics = getMetrics(isClassification, cmd.hasOption("t"));

    try {
      InitSearchModel initSearchModel =
          new InitSearchModel(
              trainFile,
              tunePath,
              extFolderPath,
              act,
              classifierList,
              asSearches,
              metrics,
              isShort ? SearchPath.SHORT : SearchPath.LONG);
      initSearchModel.initSearchModel();
    } catch (Exception ex) {
      ex.printStackTrace();
      System.exit(-1);
    }
  }

  private static List<MetricType> getMetrics(boolean isClassification, boolean hasTune) {
    return isClassification
        ? hasTune
            ? Collections.singletonList(MetricType.MCC_MEAN)
            : Collections.singletonList(MetricType.MCC_TRAIN)
        : hasTune
            ? Collections.singletonList(MetricType.MAE_MEAN)
            : Collections.singletonList(MetricType.MAE_TRAIN);
  }
}
