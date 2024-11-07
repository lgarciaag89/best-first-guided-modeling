package tomocomd.searchmodels.v3.utils;

import org.apache.commons.cli.*;

public class DefiningCMDOptions {

  public static Options getOptions() {
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

    options.addOption(
        "s",
        "short",
        false,
        "If it is set, the search will be short means that only one search will execute, and all the classification algorithm will execute in the same path, is faster but may fall into local optima");
    options.addOption("h", "help", false, "Show this help and exit");
    options.addOption("v", "version", false, "show the version and exit");
    return options;
  }

  public static CommandLine getCommandLine(String[] args) {
    Options options = getOptions();
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
    return cmd;
  }
}
