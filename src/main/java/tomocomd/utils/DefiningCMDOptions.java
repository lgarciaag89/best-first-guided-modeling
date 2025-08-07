package tomocomd.utils;

import java.util.Arrays;
import java.util.stream.Collectors;
import org.apache.commons.cli.*;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import tomocomd.classifiers.ClassifierNameEnum;
import tomocomd.searchmodels.v3.utils.MetricType;

public class DefiningCMDOptions {

  private static final Logger logger = LogManager.getLogger(DefiningCMDOptions.class);

  private DefiningCMDOptions() {
    throw new IllegalStateException("Utility class");
  }

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

    String listClass =
        Arrays.stream(ClassifierNameEnum.values())
            .map(
                classEnum -> {
                  String classifierName = classEnum.getClassifierName().toUpperCase();
                  if (classEnum.getProblemType() == MetricType.ProblemType.CLASSIFICATION) {
                    return classifierName + "(C)";
                  } else if (classEnum.getProblemType() == MetricType.ProblemType.REGRESSION) {
                    return classifierName + "(R)";
                  } else {
                    return classifierName + "(C,R)";
                  }
                })
            .collect(Collectors.joining(", "));

    Option opt =
        new Option(
            "m",
            "models",
            true,
            "Space separate list of desired strategies. The strategies are: ["
                + listClass
                + "]  where C=Classification, R=Regression.  Use \"all\" to apply all possible models");
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

    options.addOption(
        "re",
        "restart",
        true,
        "Restart an incomplete execution, receive a file with the status of the incomplete execution.");

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
      logger.error("Problems parsing command line:{}", ex.getMessage(), ex);
      System.exit(-1);
    }

    if (cmd.hasOption("h")) {
      help.printHelp("cmd", options, true);
      System.exit(0);
    } else if (cmd.hasOption("v")) {
      System.out.println(VersionUtil.getVersionInfo());
      System.exit(0);
    }

    String argsString = String.join(" ", args);

    logger.info("Command line:{} ", argsString);
    return cmd;
  }
}
