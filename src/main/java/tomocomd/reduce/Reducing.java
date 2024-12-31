package tomocomd.reduce;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import tomocomd.CSVManage;
import tomocomd.ModelingException;
import tomocomd.utils.Constants;
import tomocomd.utils.ReadData;
import tomocomd.utils.Removing;
import weka.attributeSelection.*;
import weka.core.Instances;

public class Reducing {

  public static boolean applyReduce(
      File trainFile, File tunePath, File extFolderPath, String act, boolean isClassification) {
    Instances trainData = ReadData.readData(trainFile, act, isClassification);
    Instances testData = ReadData.readData(tunePath, act, isClassification);
    List<Instances> externalTestData =
        ReadData.loadingExternalTestPath(extFolderPath, act, isClassification);

    int[] bestFirstSelected;
    try {
      bestFirstSelected = cfsBestFirst(new Instances(trainData));
    } catch (Exception e) {
      throw ModelingException.ExceptionType.REDUCE_EXCEPTION.get(e);
    }
    List<ASEvaluation> selectors =
        isClassification
            ? Arrays.asList(
                new CorrelationAttributeEval(),
                new ChiSquaredAttributeEval(),
                new InfoGainAttributeEval(),
                new ReliefFAttributeEval(),
                new SymmetricalUncertAttributeEval())
            : Arrays.asList(
                new ReliefFAttributeEval(), new CorrelationAttributeEval(), new SEAttributeEval());

    int[] positions =
        selectors.stream()
            .map(evaluator -> applyRanker(trainData, evaluator, bestFirstSelected.length - 1))
            .filter(Objects::nonNull)
            .flatMapToInt(Arrays::stream)
            .distinct()
            .sorted()
            .toArray();

    Instances filteredTrain = Removing.executeRemove(trainData, positions, true);
    CSVManage.saveDescriptorMResult(
        filteredTrain, trainFile.getAbsolutePath() + Constants.REDUCE_MARK);

    if (Objects.nonNull(testData)) {
      Instances filteredTune = Removing.executeRemove(testData, positions, true);
      CSVManage.saveDescriptorMResult(
          filteredTune, tunePath.getAbsolutePath() + Constants.REDUCE_MARK);
    }

    if (Objects.nonNull(extFolderPath)) {
      File externalFilterFolder =
          new File(extFolderPath.getAbsolutePath() + Constants.REDUCE_MARK_FOLDER);

      externalFilterFolder.mkdir();
      externalTestData.forEach(
          ext -> {
            Instances filteredExt = Removing.executeRemove(ext, positions, true);
            CSVManage.saveDescriptorMResult(
                filteredExt,
                new File(externalFilterFolder, ext.relationName() + Constants.REDUCE_MARK)
                    .getAbsolutePath());
          });
    }
    return true;
  }

  public static int[] applyRanker(Instances train, ASEvaluation evaluator, int size) {
    try {
      Ranker ranker = new Ranker();

      ranker.setOptions(new String[] {"-N", Integer.toString(size)});

      AttributeSelection asReliefFAttributeEval = new AttributeSelection();
      asReliefFAttributeEval.setEvaluator(evaluator);
      asReliefFAttributeEval.setSearch(ranker);
      asReliefFAttributeEval.setXval(false);
      asReliefFAttributeEval.SelectAttributes(train);
      return asReliefFAttributeEval.selectedAttributes();
    } catch (Exception e) {
      return null;
    }
  }

  public static int[] cfsBestFirst(Instances train) throws Exception {
    AttributeSelection asCfsSubsetEvalForw = new AttributeSelection();
    asCfsSubsetEvalForw.setEvaluator(new CfsSubsetEvalDiscretePrecision());
    BestFirst bf = new BestFirst();
    bf.setOptions(new String[] {"-D", "2"});
    asCfsSubsetEvalForw.setSearch(bf);
    asCfsSubsetEvalForw.setXval(false);
    asCfsSubsetEvalForw.SelectAttributes(train);
    return asCfsSubsetEvalForw.selectedAttributes();
  }
}
