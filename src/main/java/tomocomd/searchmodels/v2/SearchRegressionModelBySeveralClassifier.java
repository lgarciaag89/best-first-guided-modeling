/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tomocomd.searchmodels.v2;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;
import tomocomd.ModelingException;
import tomocomd.searchmodels.RegressionModelInfo;
import tomocomd.searchmodels.RegressionOptimizationParam;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;

public class SearchRegressionModelBySeveralClassifier extends ASearchModelBySeveralClassifiers {

  private static final String FORMAT = "%s,%s";
  private final RegressionOptimizationParam opt;
  private List<RegressionModelInfo> listMInfo;

  public SearchRegressionModelBySeveralClassifier(
      long mId,
      String trainPath,
      String testPath,
      List<String> externalTestPath,
      String pathToSave,
      int classAct,
      List<AbstractClassifier> cls,
      RegressionOptimizationParam oP,
      List<RegressionModelInfo> modelInfoList)
      throws ModelingException {

    super(trainPath, testPath, pathToSave, classAct, mId, externalTestPath, cls);

    opt = oP;
    listMInfo = modelInfoList;
  }

  public SearchRegressionModelBySeveralClassifier(
      long mId,
      Instances data,
      String trainPath,
      String testPath,
      List<String> externalTestPath,
      String pathToSave,
      int classAct,
      List<AbstractClassifier> cls,
      RegressionOptimizationParam oP,
      List<RegressionModelInfo> modelInfoList)
      throws ModelingException {

    super(data, trainPath, testPath, pathToSave, classAct, mId, externalTestPath, cls);

    opt = oP;
    listMInfo = modelInfoList;
  }

  public SearchRegressionModelBySeveralClassifier(
      long mId,
      Instances data,
      String trainPath,
      Instances test,
      String testPath,
      List<String> externalTestPath,
      String pathToSave,
      int classAct,
      List<AbstractClassifier> cls,
      RegressionOptimizationParam oP,
      List<RegressionModelInfo> modelInfoList)
      throws ModelingException {

    super(data, trainPath, test, testPath, pathToSave, classAct, mId, externalTestPath, cls);

    opt = oP;
    listMInfo = modelInfoList;
  }

  @Override
  protected double getEvaluation(
      Instances train, Instances internalTest, List<Instances> externalTests)
      throws ModelingException {
    Set<String> mdRegNames = getMDNames(train);

    double valReg = 0;

    for (AbstractClassifier classifier : classifiers) {

      Instances localTrain = new Instances(train);
      localTrain.setClassIndex(classAct);
      Instances localInternalTest =
          Objects.nonNull(internalTest) ? new Instances(internalTest) : null;
      if (Objects.nonNull(localInternalTest)) localInternalTest.setClassIndex(classAct);

      List<Instances> localExternalTests = Collections.emptyList();
      if (Objects.nonNull(externalTests)) {
        localExternalTests =
            externalTests.stream()
                .map(
                    data -> {
                      Instances localData = new Instances(data);
                      localData.setClassIndex(classAct);
                      return localData;
                    })
                .collect(Collectors.toList());
      }
      valReg =
          Math.max(
              valReg,
              evaluateOneClass(
                  localTrain, localInternalTest, localExternalTests, mdRegNames, classifier));
    }

    return valReg;
  }

  @Override
  protected double evaluateOneClass(
      Instances train,
      Instances internalTest,
      List<Instances> externalTests,
      Set<String> mdRegNames,
      AbstractClassifier classifier) {
    double valReg = 0;
    try {
      AbstractClassifier clasTmp = (AbstractClassifier) AbstractClassifier.makeCopy(classifier);
      if (clasTmp instanceof IBk) ((IBk) clasTmp).setKNN(Math.max(10, train.numInstances() / 10));
      clasTmp.buildClassifier(train);
      Classifier[] copies = AbstractClassifier.makeCopies(clasTmp, 2 + externalTestPath.size());

      Evaluation rfEvaTrain = new Evaluation(train);
      rfEvaTrain.crossValidateModel(copies[1], train, 10, new Random(1));
      double rST = rfEvaTrain.correlationCoefficient();
      double maeT = rfEvaTrain.meanAbsoluteError();
      double rmseT = rfEvaTrain.rootMeanSquaredError();

      double rSE;
      double maeE;
      double rmseE;

      if (internalTest != null) {
        Evaluation rfEvaTest = new Evaluation(train);
        rfEvaTest.evaluateModel(copies[0], internalTest);
        rSE = rfEvaTest.correlationCoefficient();
        maeE = rfEvaTest.meanAbsoluteError();
        rmseE = rfEvaTest.rootMeanSquaredError();
      } else {
        maeE = maeT;
        rSE = rST;
        rmseE = rmseT;
      }

      double q2ext = rSE * rSE;
      double q2 = rST * rST;
      double prom = (q2 + q2ext) / 2;
      switch (opt) {
        case MaeExt:
          valReg = maeE;
          break;
        case MaeTrain:
          valReg = maeT;
          break;
        case Q2Ext:
          valReg = q2ext;
          break;
        case Q2train:
          valReg = q2;
          break;
        case Mean:
          valReg = prom;
          break;
      }

      String resExternal = evaluateExternals(train, externalTests, copies);

      String res;
      if (internalTest != null) {
        res =
            buildLineInfoModel(
                q2, maeT, rmseT, q2ext, maeE, rmseE, resExternal, mdRegNames, clasTmp);
      } else {
        res = buildLineInfoModel(q2, maeT, rmseT, 0, 0, 0, resExternal, mdRegNames, clasTmp);
      }

      addLineModelToFile(res);
    } catch (Exception ex) {
      throw ModelingException.ExceptionType.BUILDING_MODEL_EXCEPTION.get(
          "Problems building and evaluating the model ", ex);
    }
    return valReg;
  }

  private void addLineModelToFile(String res) throws ModelingException {

    String line = "";
    if (!(new File(pathToSave).exists())) {
      String head =
          testPath == null
              ? "classifier,id,size,Q2_CV,MAE_CV, RMSE_CV"
              : "classifier,id,size,Q2_CV,MAE_CV, RMSE_CV,Q2_Tune,MAE_Tune, RMSE_TUNE";
      String extHead = "";
      if (!externalTestPath.isEmpty()) {
        for (String nameExt : externalTestPath) {
          String singleName = new File(nameExt).getName();
          String headPart = String.format("Q2_%s,MAE_%s", singleName, singleName);
          if (extHead.isEmpty()) {
            extHead = headPart;
          } else {
            extHead = String.format(FORMAT, extHead, headPart);
          }
        }
      }
      line =
          extHead.isEmpty()
              ? String.format("%s,desc%n", head)
              : String.format("%s,%s,desc%n", head, extHead);
    }
    line += res;

    try (FileWriter fw = new FileWriter(pathToSave, true);
        BufferedWriter w = new BufferedWriter(fw)) {
      w.write(line + "\n");
    } catch (IOException ex) {
      throw ModelingException.ExceptionType.CSV_FILE_WRITING_EXCEPTION.get(
          "Problems saving models results on file " + pathToSave, ex);
    }
  }

  private String buildLineInfoModel(
      double q2,
      double mae,
      double rmse,
      double q2Ext,
      double maeExt,
      double rmseE,
      String resExternal,
      Set<String> mdNames,
      AbstractClassifier clasTmp)
      throws ModelingException {
    String clasName = getClassifierName(clasTmp);

    String res =
        String.format("%s,%d,%d,%.5f,%.5f,%.5f", clasName, modelId, mdNames.size(), q2, mae, rmse);

    if (testPath != null) res = String.format("%s,%.5f,%.5f,%.5f", res, q2Ext, maeExt, rmseE);

    if (!resExternal.isEmpty()) {
      res = String.format(FORMAT, res, resExternal);
    }

    res = String.format(FORMAT, res, mdNames.toString().replace(",", " "));

    listMInfo.add(
        new RegressionModelInfo(
            clasTmp,
            ++modelId,
            mdNames.size(),
            q2,
            mae,
            rmse,
            q2Ext,
            maeExt,
            rmseE,
            mdNames.toArray(new String[0]),
            "Ext_" + clasName,
            ""));

    return res;
  }

  private String evaluateExternals(
      Instances instTrain, List<Instances> extInst, Classifier[] copies) throws ModelingException {
    String resExternal = "";
    int t = 0;
    for (Instances ext : extInst) {
      Evaluation rfEvaExt;
      double rSE;
      double maeE;
      try {
        rfEvaExt = new Evaluation(instTrain);
        rfEvaExt.evaluateModel(copies[t + 2], ext);
      } catch (Exception e) {
        throw ModelingException.ExceptionType.CLASSIFIER_LOAD_EXCEPTION.get(
            "Problems evaluating model on external sets", e);
      }

      try {
        rSE = rfEvaExt.correlationCoefficient();
      } catch (Exception e) {
        throw ModelingException.ExceptionType.CLASSIFIER_LOAD_EXCEPTION.get(e);
      }
      maeE = rfEvaExt.meanAbsoluteError();
      if (resExternal.isEmpty()) {
        resExternal = String.format("%.5f,%.5f", rSE * rSE, maeE);
      } else {
        resExternal = String.format("%s,%.5f,%.5f", resExternal, rSE * rSE, maeE);
      }
    }
    return resExternal;
  }
}
