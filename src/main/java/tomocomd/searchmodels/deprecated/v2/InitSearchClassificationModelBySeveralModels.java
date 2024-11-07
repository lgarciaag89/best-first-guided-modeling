package tomocomd.searchmodels.deprecated.v2;

import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import tomocomd.ModelingException;
import tomocomd.searchmodels.deprecated.AInitSearchModel;
import tomocomd.searchmodels.deprecated.ClassificationModelInfo;
import tomocomd.searchmodels.deprecated.ClassificationOptimizationParam;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.AttributeSelection;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

public class InitSearchClassificationModelBySeveralModels extends AInitSearchModel {

  private final List<ClassificationOptimizationParam> params;

  public InitSearchClassificationModelBySeveralModels(
      String train,
      String tune,
      String folderExtSdf,
      String act,
      List<AbstractClassifier> classifierList,
      List<ASSearch> searchList,
      List<ClassificationOptimizationParam> params) {
    super(train, tune, folderExtSdf, act, classifierList, searchList);
    this.params = new LinkedList<>(params);
  }

  public InitSearchClassificationModelBySeveralModels(
      Instances train,
      String csvFile,
      Instances test,
      String tuneSdf,
      String folderExtSdf,
      String act,
      List<AbstractClassifier> classifierList,
      List<ASSearch> searchList,
      List<ClassificationOptimizationParam> params) {
    super(train, csvFile, test, tuneSdf, folderExtSdf, act, classifierList, searchList);
    this.params = new LinkedList<>(params);
  }

  @Override
  public void startSearchModel() throws ModelingException {

    List<String> extInsts = loadingExternalTestPath();
    long cantModels = 0;
    Instances dataTrainThread = new Instances(trainInstances);
    Instances dataTestThread = Objects.isNull(tuneInstances) ? null : new Instances(tuneInstances);
    List<ClassificationModelInfo> models = new LinkedList<>();
    for (ASSearch aSSearch : searchList) {
      for (ClassificationOptimizationParam opt : params) {
        try {
          selectionClassificationOne(
              dataTrainThread, dataTestThread, extInsts, opt, aSSearch, cantModels, models);
          cantModels += models.size();
        } catch (Exception ex) {
          throw ModelingException.ExceptionType.BUILDING_MODEL_EXCEPTION.get(
              String.format("Problems building models from %s dataset", csvFile), ex);
        }
      }
    }
  }

  private void selectionClassificationOne(
      Instances data,
      Instances test,
      List<String> externals,
      ClassificationOptimizationParam opt,
      ASSearch aSSearch,
      long idModels,
      List<ClassificationModelInfo> models)
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

    SearchClassificationModelBySeveralClassifier search =
        Objects.nonNull(test)
            ? new SearchClassificationModelBySeveralClassifier(
                idModels,
                data,
                csvFile,
                inst,
                tuneSdf,
                externals,
                nameFile,
                cIdx,
                classifierList,
                opt,
                models)
            : new SearchClassificationModelBySeveralClassifier(
                idModels,
                data,
                csvFile,
                tuneSdf,
                externals,
                nameFile,
                cIdx,
                classifierList,
                opt,
                models);

    asSubset.setSearch(aSSearch);
    asSubset.setEvaluator(search);
    asSubset.setXval(false);

    Instances redTrain = new Instances(data, 0, 1);

    try {
      asSubset.SelectAttributes(redTrain);
      asSubset.selectedAttributes();
    } catch (Exception ex) {
      throw ModelingException.ExceptionType.BUILDING_MODEL_EXCEPTION.get(
          "Problems building classification models", ex);
    }
  }
}
