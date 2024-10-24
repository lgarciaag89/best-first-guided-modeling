package tomocomd.searchmodels.v2;

import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import tomocomd.ModelingException;
import tomocomd.searchmodels.AInitSearchModel;
import tomocomd.searchmodels.RegressionModelInfo;
import tomocomd.searchmodels.RegressionOptimizationParam;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.AttributeSelection;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

public class InitSearchRegressionModelBySeveralModels extends AInitSearchModel {
  private final List<RegressionOptimizationParam> params;

  public InitSearchRegressionModelBySeveralModels(
      String csvFile,
      String tuneSdf,
      String folderExtSdf,
      String act,
      List<AbstractClassifier> classifierList,
      List<ASSearch> searchList,
      List<RegressionOptimizationParam> params) {
    super(csvFile, tuneSdf, folderExtSdf, act, classifierList, searchList);
    this.params = new LinkedList<>(params);
  }

  public InitSearchRegressionModelBySeveralModels(
      Instances data,
      String csvFile,
      String tuneSdf,
      String folderExtSdf,
      String act,
      List<AbstractClassifier> classifierList,
      List<ASSearch> searchList,
      List<RegressionOptimizationParam> params) {
    super(data, csvFile, tuneSdf, folderExtSdf, act, classifierList, searchList);
    this.params = new LinkedList<>(params);
  }

  public InitSearchRegressionModelBySeveralModels(
      Instances train,
      String csvFile,
      Instances test,
      String tuneSdf,
      String folderExtSdf,
      String act,
      List<AbstractClassifier> classifierList,
      List<ASSearch> searchList,
      List<RegressionOptimizationParam> params) {
    super(train, csvFile, test, tuneSdf, folderExtSdf, act, classifierList, searchList);
    this.params = new LinkedList<>(params);
  }

  @Override
  public void startSearchModel() throws ModelingException {
    List<String> extInsts = loadingExternalTestPath();
    Instances dataTrainThread = new Instances(trainInstances);
    Instances dataTestThread = Objects.isNull(tuneInstances) ? null : new Instances(tuneInstances);

    List<RegressionModelInfo> models = new LinkedList<>();

    for (ASSearch aSSearch : searchList) {
      for (RegressionOptimizationParam opt : params) {
        try {
          selectionRegressionOneData(
              dataTrainThread, dataTestThread, extInsts, opt, aSSearch, 0, models);
        } catch (Exception ex) {
          throw ModelingException.ExceptionType.BUILDING_MODEL_EXCEPTION.get(
              String.format("Problems building models from %s dataset", csvFile), ex);
        }
      }
    }
  }

  private void selectionRegressionOneData(
      Instances data,
      Instances test,
      List<String> extInsts,
      RegressionOptimizationParam opt,
      ASSearch aSSearch,
      long modelId,
      List<RegressionModelInfo> models)
      throws ModelingException {

    Instances inst = null;
    int cIdx = data.classIndex();
    if (test != null) {
      inst = new Instances(test);
    }

    String nameFile =
        String.format(
            "%s_models_%s_%s.csv", csvFile, aSSearch.getClass().getSimpleName(), opt.toString());
    AttributeSelection asSubset = new AttributeSelection();

    SearchRegressionModelBySeveralClassifier search =
        Objects.nonNull(test)
            ? new SearchRegressionModelBySeveralClassifier(
                modelId,
                data,
                csvFile,
                inst,
                tuneSdf,
                extInsts,
                nameFile,
                cIdx,
                classifierList,
                opt,
                models)
            : new SearchRegressionModelBySeveralClassifier(
                modelId,
                data,
                csvFile,
                tuneSdf,
                extInsts,
                nameFile,
                cIdx,
                classifierList,
                opt,
                models);

    asSubset.setSearch(aSSearch);
    asSubset.setEvaluator(search);
    asSubset.setXval(false);

    Instances redTrain = new Instances(data);

    try {
      asSubset.SelectAttributes(redTrain);
      asSubset.selectedAttributes();
    } catch (Exception ex) {
      throw ModelingException.ExceptionType.BUILDING_MODEL_EXCEPTION.get(
          "Problems building regression models", ex);
    }
  }
}
