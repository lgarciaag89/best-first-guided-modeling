package tomocomd.utils;

import org.apache.commons.cli.*;

public class DefiningCMDOptions {

  public static Options getOptions() {
    Options options = new Options();

    options.addOption("t", "train", true, "Input training dataset in CSV format.");
    options.addOption("p", "test", true, "input, test dataset, csv format");
    options.addOption(
        "x", "external", true, "External folder with several additional datasets in CSV format.");
    options.addOption(
        "e",
        "endpoint",
        true,
        "Target property, specifies the name of the variable to be used as the target.");
    options.addOption(
        "c",
        "classification",
        false,
        "Specifies that the problem is a classification problem. If it's a regression problem, do not set this option.");

    Option opt =
        new Option(
            "m",
            "models",
            true,
            "Space separate list of desired strategies. The strategies are: "
                + "[KNN(C,R), RandomForest(C,R), Adaboost(C), AdditiveRegression(R), BayesNet(C), LogitBoost(C), "
                + "RandomCommittee(C,R), SMO-PolyKernel(C,R), SMO-Puk(C,R), LinerRegression(R), Gaussian(R), "
                + "Bagging-SMO(C,R),  Bagging-KNN(C,R)], where C=Classification, R=Regression.  Use \"all\" to apply all possible models");
    opt.setOptionalArg(false);
    opt.setArgs(Option.UNLIMITED_VALUES);
    options.addOption(opt);

    options.addOption(
        "s",
        "short",
        false,
        "If set, the search will be faster but may fall into local optima. Only one search will execute, and all classification algorithms will execute along the same path.");
    options.addOption("h", "help", false, "Displays this help message and exits.");
    options.addOption("v", "version", false, "Displays the version of the program and exits.");

    options.addOption(
        "f",
        "filter",
        false,
        "Execute filter operations (e.g., Shannon entropy (-se), Pearson correlation (-r)).");
    options.addOption(
        "pt",
        "pearson-threshold",
        true,
        "Pearson correlation threshold for eliminating highly correlated attributes.");
    options.addOption(
        "se",
        "se-threshold",
        true,
        "Shannon entropy threshold for reducing the number of attributes.");
    options.addOption("r", "reduce", false, "Reduces the number of attributes.");
    options.addOption("o", "reorder", false, "Reverses the order of the attributes.");

    return options;
  }

  public static CommandLine getCommandLine(String[] args) {
    Options options = getOptions();
    CommandLineParser parser = new DefaultParser();
    CommandLine cmd = null;
    HelpFormatter help = new HelpFormatter();
    try {
      cmd = parser.parse(options, args);
    } catch (ParseException ex) {
      help.printHelp("cmd", options, true);
      System.err.println("Problems parsing command line:" + ex.getMessage());
      System.exit(-1);
    }

    if (cmd.hasOption("h")) {
      help.printHelp("cmd", options, true);
      System.exit(0);
    } else if (cmd.hasOption("v")) {
      System.out.println(VersionUtil.getVersionInfo());
      System.exit(0);
    }
    System.out.println("Command line: " + String.join(" ", args));
    return cmd;
  }
}
