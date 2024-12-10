package tomocomd;

import java.io.File;
import java.util.Objects;
import org.apache.commons.cli.*;
import tomocomd.filters.ApplyFilter;
import tomocomd.reduce.Reducing;
import tomocomd.searchmodels.v3.BuildModels;
import tomocomd.utils.Constants;
import tomocomd.utils.DefiningCMDOptions;

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

    if (cmd.hasOption("f")) {
      ApplyFilter filters = new ApplyFilter();
      if(filters.initFilter(trainFile, tunePath, extFolderPath, act, cmd)) {
        trainFile = new File(trainFile.getAbsolutePath() + Constants.FILTER_MARK);
        if (Objects.nonNull(tunePath)) {
          tunePath = new File(tunePath.getAbsolutePath() + Constants.FILTER_MARK);
        }
        if (Objects.nonNull(extFolderPath)) {
          extFolderPath = new File(extFolderPath.getAbsolutePath() + Constants.FILTER_MARK_FOLDER);
        }
      }
    }

    if (cmd.hasOption("r")) {
      if(Reducing.applyReduce(trainFile, tunePath, extFolderPath, act, cmd.hasOption("c"))){
        trainFile = new File(trainFile.getAbsolutePath() + Constants.REDUCE_MARK);
        if (Objects.nonNull(tunePath)) {
          tunePath = new File(tunePath.getAbsolutePath() + Constants.REDUCE_MARK);
        }
        if (Objects.nonNull(extFolderPath)) {
          extFolderPath = new File(extFolderPath.getAbsolutePath() + Constants.REDUCE_MARK_FOLDER);
        }
      }
    }

    if (cmd.hasOption("m")) {
      BuildModels.buildModels(trainFile, tunePath, extFolderPath, act, cmd);
    }
  }
}
