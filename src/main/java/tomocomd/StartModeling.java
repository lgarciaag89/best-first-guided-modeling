package tomocomd;

import java.io.File;
import java.util.*;
import org.apache.commons.cli.*;
import tomocomd.searchmodels.*;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.BestFirst;
import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

/** Hello world! */
public class StartModeling {
  public static void main(String[] args) {
    parseOptions(args);
  }

  private static void parseOptions(String[] args) {

    Options options = new Options();

    options.addOption("t", "train", true, "input, train dataset");
    options.addOption("p", "test", true, "input, test dataset, csv format");
    options.addOption(
        "x", "external", true, "input, external folder with several external datasets, csv format");
    options.addOption("e", "endpoint", true, "property target");
    options.addOption(
        "c",
        "classification",
        false,
        "it is a classification problem, if it is a regression problem not set this option");
    Option opt =
        new Option(
            "m",
            "models",
            true,
            "List with the desirable strategies, "
                + "[KNN(C,R),RandomForest(C,R),Adaboost(C),BayesNet(C),Gradient(C),J48(C),Logistic(C), LogitBoost(C),"
                + "SimpleLogistic(C), MultiBost(C), NaiveBayes(C),RacedIncrementalLogitBoost(C) RandomCommittee(C,R),"
                + " RandomTree(C), SMO(C,R), SVM(C), MultilayerPerceptron(R), LinerRegression(R)], all indicates all the possibles strategies");
    opt.setOptionalArg(false);
    opt.setArgs(Option.UNLIMITED_VALUES);
    options.addOption(opt);

    options.addOption("h", "help", false, "Show this help and exit");
    options.addOption("v", "version", false, "show the version and exit");

    CommandLineParser parser = new DefaultParser();
    CommandLine cmd = null;
    HelpFormatter help = new HelpFormatter();
    try {
      cmd = parser.parse(options, args);
      System.out.println("Command line: " + String.join(" ", args));
    } catch (ParseException ex) {
      help.printHelp("cmd", options, true);
      System.err.println("Problems parsing command line:" + ex.getMessage());
      System.exit(-1);
    }

    if (cmd.hasOption("h")) {
      help.printHelp("cmd", options, true);
      System.exit(0);
    } else if (cmd.hasOption("v")) {
      System.out.println("build-tomocomd-models 1.0");
      System.exit(0);
    }

    File trainFile = null;
    try {
      trainFile = new File(cmd.getOptionValue("t"));
    } catch (Exception ex) {
      System.err.println("Error loading train dataset:" + ex.getMessage());
      System.exit(-1);
    }

    System.out.println("Train file: " + trainFile.getAbsolutePath());

    String act = "";
    if (cmd.hasOption("e")) {
      act = cmd.getOptionValue("e");
    }

    if (act.isEmpty()) {
      System.err.println("It is important define endpoint class");
      System.exit(-1);
    }

    System.out.println("Endpoint: " + act);

    File tunePath = null;
    Instances tune = null;
    if (cmd.hasOption("p")) {
      tunePath = new File(cmd.getOptionValue("p"));
      tune = CSVManage.loadCSV(tunePath.getAbsolutePath());
      tune.setClassIndex(0);
      System.out.println("Tune file: " + tunePath.getAbsolutePath());
    }

    String extFolderPath = null;
    if (cmd.hasOption("x")) {
      if (!new File(cmd.getOptionValue("x")).exists()) {
        System.err.println("External folder does not exist");
        System.exit(-1);
      }
      extFolderPath = new File(cmd.getOptionValue("x")).getAbsolutePath();
      System.out.println("External folder: " + extFolderPath);
    }

    boolean isClassification = cmd.hasOption('c');

    List<AbstractClassifier> classifierList =
        cmd.hasOption("m")
            ? BuildClassifierList.getclassifierList(cmd.getOptionValues("m"), isClassification)
            : Collections.emptyList();

    Instances data = CSVManage.loadCSV(trainFile.getAbsolutePath());
    data.setClassIndex(0);

    ArrayList<ASSearch> asSearches = new ArrayList<>(Arrays.asList(new BestFirst()));

    try {
      if (cmd.hasOption('c')) {
        List<ClassificationOptimizationParam> params =
            Objects.isNull(tunePath)
                ? new ArrayList<>(
                    Collections.singletonList(ClassificationOptimizationParam.MCC_TRAIN))
                : new ArrayList<>(
                    Collections.singletonList(ClassificationOptimizationParam.MCC_MEAN));

        Attribute attribute = data.attribute(act);
        NumericToNominal filter = new NumericToNominal();
        filter.setAttributeIndicesArray(new int[] {attribute.index()});
        filter.setInputFormat(data);
        Instances newData = Filter.useFilter(data, filter);
        Instances newTest = Objects.nonNull(tune) ? Filter.useFilter(tune, filter) : null;

        String pathTune = Objects.nonNull(tunePath) ? tunePath.getAbsolutePath() : null;
        InitSearchClassificationModel initSearchClassificationModel =
            new InitSearchClassificationModel(
                newData,
                trainFile.getAbsolutePath(),
                newTest,
                pathTune,
                extFolderPath,
                act,
                classifierList,
                asSearches,
                params);

        initSearchClassificationModel.startSearchModel();

      } else {
        List<RegressionOptimizationParam> params =
            Objects.isNull(tunePath)
                ? new ArrayList<>(Collections.singletonList(RegressionOptimizationParam.MaeTrain))
                : new ArrayList<>(Collections.singletonList(RegressionOptimizationParam.Mean));
        String pathTune = Objects.nonNull(tunePath) ? tunePath.getAbsolutePath() : null;
        InitSearchRegressionModel initSearchRegressionModel =
            new InitSearchRegressionModel(
                data,
                trainFile.getAbsolutePath(),
                tune,
                pathTune,
                extFolderPath,
                act,
                classifierList,
                asSearches,
                params);

        initSearchRegressionModel.startSearchModel();
      }
    } catch (Exception ex) {
      ex.printStackTrace();
      System.exit(-1);
    }
  }
}
