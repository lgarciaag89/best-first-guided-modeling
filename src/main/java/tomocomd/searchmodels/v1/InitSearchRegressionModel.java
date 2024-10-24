package tomocomd.searchmodels.v1;

import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import tomocomd.CSVManage;
import tomocomd.ModelingException;
import tomocomd.searchmodels.AInitSearchModel;
import tomocomd.searchmodels.RegressionModelInfo;
import tomocomd.searchmodels.RegressionOptimizationParam;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.AttributeSelection;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

public class InitSearchRegressionModel extends AInitSearchModel {
  private final List<RegressionOptimizationParam> params;

  public InitSearchRegressionModel(
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

  public InitSearchRegressionModel(
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

  public InitSearchRegressionModel(
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
    for (AbstractClassifier clas : classifierList) {
      for (ASSearch aSSearch : searchList) {
        for (RegressionOptimizationParam opt : params) {
          try {
            selectionRegressionOneData(
                dataTrainThread,
                csvFile,
                dataTestThread,
                tuneSdf,
                extInsts,
                opt,
                clas,
                aSSearch,
                0,
                models);
          } catch (Exception ex) {
            throw ModelingException.ExceptionType.BUILDING_MODEL_EXCEPTION.get(
                String.format("Problems building models from %s dataset", csvFile), ex);
          }
        }
      }
    }
  }

  private static void selectionRegressionOne(
      String pathTrain,
      String pathTest,
      List<String> extInsts,
      String act,
      RegressionOptimizationParam opt,
      AbstractClassifier classifier,
      ASSearch aSSearch,
      long modelId,
      List<RegressionModelInfo> models)
      throws ModelingException {

    Instances inst;
    Instances train = CSVManage.loadCSV(pathTrain);

    int cIdx = 0;
    int numAtt = train.numAttributes();
    for (int i = 0; i < numAtt; i++) {
      if (train.attribute(i).name().equals(act)) {
        cIdx = i;
      }
    }
    train.setClassIndex(cIdx);

    if (pathTest != null) {
      if (!pathTest.isEmpty()) {
        inst = new Instances(CSVManage.loadCSV(pathTest));
        inst.setClassIndex(cIdx);
      }
    }

    String nameFile =
        String.format(
            "%s_models_%s_%s_%s.csv",
            pathTrain,
            classifier.getClass().getSimpleName(),
            aSSearch.getClass().getSimpleName(),
            opt.toString());
    AttributeSelection asSubset = new AttributeSelection();
    SearchRegressionModelByOneClasifier search =
        new SearchRegressionModelByOneClasifier(
            modelId, pathTrain, pathTest, extInsts, nameFile, cIdx, classifier, opt, models);

    asSubset.setSearch(aSSearch);
    asSubset.setEvaluator(search);
    asSubset.setXval(false);

    Instances redTrain = new Instances(train, 0, 1);

    try {
      asSubset.SelectAttributes(redTrain);
      asSubset.selectedAttributes();
    } catch (Exception ex) {
      throw ModelingException.ExceptionType.BUILDING_MODEL_EXCEPTION.get(
          "Problems building regression models", ex);
    }
  }

  private static void selectionRegressionOneData(
      Instances data,
      String pathTrain,
      Instances test,
      String pathTest,
      List<String> extInsts,
      RegressionOptimizationParam opt,
      AbstractClassifier classifier,
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
            "%s_models_%s_%s_%s.csv",
            pathTrain,
            classifier.getClass().getSimpleName(),
            aSSearch.getClass().getSimpleName(),
            opt.toString());
    AttributeSelection asSubset = new AttributeSelection();

    SearchRegressionModelByOneClasifier search =
        Objects.nonNull(test)
            ? new SearchRegressionModelByOneClasifier(
                modelId,
                data,
                pathTrain,
                inst,
                pathTest,
                extInsts,
                nameFile,
                cIdx,
                classifier,
                opt,
                models)
            : new SearchRegressionModelByOneClasifier(
                modelId,
                data,
                pathTrain,
                pathTest,
                extInsts,
                nameFile,
                cIdx,
                classifier,
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
