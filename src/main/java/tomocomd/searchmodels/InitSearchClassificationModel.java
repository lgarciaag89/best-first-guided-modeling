package tomocomd.searchmodels;

import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import tomocomd.CSVManage;
import tomocomd.ModelingException;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.AttributeSelection;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

public class InitSearchClassificationModel extends AInitSearchModel {

  private final List<ClassificationOptimizationParam> params;

  public InitSearchClassificationModel(
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

  public InitSearchClassificationModel(
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
    for (AbstractClassifier clas : classifierList) {
      for (ASSearch aSSearch : searchList) {
        for (ClassificationOptimizationParam opt : params) {
          try {
            selectionClassificationOne(
                dataTrainThread,
                csvFile,
                dataTestThread,
                tuneSdf,
                extInsts,
                opt,
                clas,
                aSSearch,
                cantModels,
                models);
            cantModels += models.size();
          } catch (Exception ex) {
            throw ModelingException.ExceptionType.BUILDING_MODEL_EXCEPTION.get(
                String.format("Problems building models from %s dataset", csvFile), ex);
          }
        }
      }
    }
  }

  private static void selectionClassificationOne(
      String pathCSV,
      String test,
      List<String> externals,
      String act,
      ClassificationOptimizationParam opt,
      AbstractClassifier classifier,
      ASSearch aSSearch,
      long idModels,
      List<ClassificationModelInfo> models)
      throws ModelingException {

    Instances inst;
    Instances train;

    train = CSVManage.loadCSV(pathCSV);
    train.setClassIndex(0);

    int cIdx = 0;
    int numAtt = train.numAttributes();
    for (int i = 0; i < numAtt; i++) {
      if (train.attribute(i).name().equals(act)) {
        cIdx = i;
      }
    }
    train.setClassIndex(cIdx);

    if (test != null) {
      if (!test.isEmpty()) {
        inst = new Instances(CSVManage.loadCSV(test));
        inst.setClassIndex(cIdx);
      }
    }

    AttributeSelection asSubset = new AttributeSelection();
    SearchClassificationModel search =
        new SearchClassificationModel(
            idModels,
            pathCSV,
            test,
            externals,
            pathCSV + "_models.csv",
            cIdx,
            classifier,
            opt,
            models);

    asSubset.setSearch(aSSearch);
    asSubset.setEvaluator(search);
    asSubset.setXval(false);

    Instances redTrain = new Instances(train, 0, 1);

    try {
      asSubset.SelectAttributes(redTrain);
      asSubset.selectedAttributes();
    } catch (Exception ex) {
      throw ModelingException.ExceptionType.BUILDING_MODEL_EXCEPTION.get(
          "Problems building classification models", ex);
    }
  }

  private static void selectionClassificationOne(
      Instances data,
      String pathTrain,
      Instances test,
      String pathTest,
      List<String> externals,
      ClassificationOptimizationParam opt,
      AbstractClassifier classifier,
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
            "%s_models_%s_%s_%s.csv",
            pathTrain,
            classifier.getClass().getSimpleName(),
            aSSearch.getClass().getSimpleName(),
            opt.toString());
    AttributeSelection asSubset = new AttributeSelection();

    SearchClassificationModel search =
        Objects.nonNull(test)
            ? new SearchClassificationModel(
                idModels,
                data,
                pathTrain,
                inst,
                pathTest,
                externals,
                nameFile,
                cIdx,
                classifier,
                opt,
                models)
            : new SearchClassificationModel(
                idModels,
                data,
                pathTrain,
                pathTest,
                externals,
                nameFile,
                cIdx,
                classifier,
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
