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
    this.filters.put(FilterType.R, new PearsonFiltering());
    this.filters.put(FilterType.SE, new SEFiltering());
  }

  public boolean initFilter(
      File trainFile, File tunePath, File extFolderPath, String act, CommandLine cmd) {
    boolean isClassification = cmd.hasOption('c');
    Instances trainData = ReadData.readTrainData(trainFile, act, isClassification);
    Instances testData = ReadData.readTuneData(tunePath, trainData.classIndex(), isClassification);
    List<Instances> externalTestData =
        ReadData.loadingExternalTestPath(extFolderPath, trainData.classIndex(), isClassification);

    Map<FilterType, Double> thresholds = new HashMap<>();
    if (cmd.hasOption("pt")) {
      thresholds.put(FilterType.R, Double.parseDouble(cmd.getOptionValue("pt")));
    }
    if (cmd.hasOption("se")) {
      thresholds.put(FilterType.SE, Double.parseDouble(cmd.getOptionValue("se")));
    }

    if (thresholds.isEmpty()) {
      return false;
    }

    int[] toRemoveArray =
        IntStream.range(0, trainData.numAttributes())
            .filter(i -> i != trainData.classIndex())
            .filter(
                i ->
                    thresholds.entrySet().stream()
                        .map(
                            entry ->
                                filters
                                    .get(entry.getKey())
                                    .passFilter(trainData, i, entry.getValue()))
                        .reduce((a, b) -> !a || !b)
                        .orElse(false))
            .toArray();

    Instances filteredTrain = Removing.executeRemove(trainData, toRemoveArray, false);
    CSVManage.saveDescriptorMResult(
        filteredTrain, trainFile.getAbsolutePath() + Constants.FILTER_MARK);

    if (Objects.nonNull(testData)) {
      Instances filteredTune = Removing.executeRemove(testData, toRemoveArray, false);
      CSVManage.saveDescriptorMResult(
          filteredTune, tunePath.getAbsolutePath() + Constants.FILTER_MARK);
    }

    File externalFilterFolder =
        new File(extFolderPath.getAbsolutePath() + Constants.FILTER_MARK_FOLDER);
    externalFilterFolder.mkdir();
    externalTestData.forEach(
        ext -> {
          Instances filteredExt = Removing.executeRemove(ext, toRemoveArray, false);
          CSVManage.saveDescriptorMResult(
              filteredExt,
              new File(externalFilterFolder, ext.relationName() + Constants.FILTER_MARK)
                  .getAbsolutePath());
        });

    return true;
  }
}
