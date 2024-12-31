package tomocomd.filters;

import java.io.File;
import java.util.*;
import java.util.stream.IntStream;
import org.apache.commons.cli.CommandLine;
import tomocomd.CSVManage;
import tomocomd.utils.Constants;
import tomocomd.utils.ReadData;
import tomocomd.utils.Removing;
import weka.core.Instances;

public class ApplyFilter {

  private final Map<FilterType, Filter> filters;

  public ApplyFilter() {
    this.filters = new LinkedHashMap<>();
    this.filters.put(FilterType.SE, new SEFiltering());
  }

  public boolean initFilter(
      File trainFile, File tunePath, File extFolderPath, String act, CommandLine cmd) {
    boolean isClassification = cmd.hasOption('c');
    Instances trainData = ReadData.readData(trainFile, act, isClassification);

    Map<FilterType, Double> thresholds = initializeThresholds(cmd);
    if (thresholds.isEmpty()) return false;

    int[] toRemoveArray = filterAttributes(trainData, thresholds);
    Instances filteredTrain = Removing.executeRemove(trainData, toRemoveArray, false);
    ReadData.setClassIndex(filteredTrain, act);

    int[] toRemoveArrayCorr =
        cmd.hasOption("pt")
            ? PearsonFiltering.getCorrelated(
                filteredTrain, Double.parseDouble(cmd.getOptionValue("pt")))
            : new int[0];

    filteredTrain = Removing.executeRemove(filteredTrain, toRemoveArrayCorr, false);

    saveFilteredResults(
        filteredTrain,
        trainFile,
        tunePath,
        extFolderPath,
        toRemoveArray,
        toRemoveArrayCorr,
        act,
        isClassification);
    return true;
  }

  private void saveFilteredResults(
      Instances filteredTrain,
      File trainFile,
      File tunePath,
      File extFolderPath,
      int[] toRemoveArray,
      int[] toRemoveArrayCorr,
      String act,
      boolean isClassification) {

    Instances testData = ReadData.readData(tunePath, act, isClassification);
    List<Instances> externalTestData =
        ReadData.loadingExternalTestPath(extFolderPath, act, isClassification);

    CSVManage.saveDescriptorMResult(
        filteredTrain, trainFile.getAbsolutePath() + Constants.FILTER_MARK);

    if (testData != null) {
      Instances filteredTune = Removing.executeRemove(testData, toRemoveArray, false);
      filteredTune = Removing.executeRemove(filteredTune, toRemoveArrayCorr, false);
      CSVManage.saveDescriptorMResult(
          filteredTune, tunePath.getAbsolutePath() + Constants.FILTER_MARK);
    }
    if (extFolderPath != null) {
      File externalFilterFolder =
          new File(extFolderPath.getAbsolutePath() + Constants.FILTER_MARK_FOLDER);
      externalFilterFolder.mkdir();
      externalTestData.forEach(
          ext -> {
            Instances filteredExt = Removing.executeRemove(ext, toRemoveArray, false);
            filteredExt = Removing.executeRemove(filteredExt, toRemoveArrayCorr, false);
            CSVManage.saveDescriptorMResult(
                filteredExt,
                new File(externalFilterFolder, ext.relationName() + Constants.FILTER_MARK)
                    .getAbsolutePath());
          });
    }
  }

  private int[] filterAttributes(Instances trainData, Map<FilterType, Double> thresholds) {
    return IntStream.range(0, trainData.numAttributes())
        .filter(i -> i != trainData.classIndex())
        .filter(
            i ->
                thresholds.entrySet().stream()
                    .noneMatch(
                        entry ->
                            filters.get(entry.getKey()).passFilter(trainData, i, entry.getValue())))
        .toArray();
  }

  private Map<FilterType, Double> initializeThresholds(CommandLine cmd) {
    Map<FilterType, Double> thresholds = new HashMap<>();
    if (cmd.hasOption("se")) {
      thresholds.put(FilterType.SE, Double.parseDouble(cmd.getOptionValue("se")));
    } else if (cmd.hasOption("pt")) {
      thresholds.put(FilterType.SE, 0.0);
    }
    return thresholds;
  }

  public static int[] mergeWithoutDuplicates(int[] array1, int[] array2) {
    return IntStream.concat(Arrays.stream(array1), Arrays.stream(array2)).distinct().toArray();
  }
}
