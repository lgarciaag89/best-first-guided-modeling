/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tomocomd.searchmodels;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Random;
import java.util.Set;
import tomocomd.ModelingException;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;

/** @author potter */
public class SearchClassificationModel extends ASearchModel {

  private static final String FORMAT_HEAD = "%s,%s";
  private final ClassificationOptimizationParam opt;
  private List<ClassificationModelInfo> listMInfo;

  public SearchClassificationModel(
      long mId,
      String trainPath,
      String testPath,
      List<String> externalTestPath,
      String pathToSave,
      int classAct,
      AbstractClassifier cls,
      ClassificationOptimizationParam oP,
      List<ClassificationModelInfo> listClassModelInfo)
      throws ModelingException {
    super(trainPath, testPath, pathToSave, classAct, mId, externalTestPath, cls);

    listMInfo = listClassModelInfo;
    opt = oP;
  }

  public SearchClassificationModel(
      long mId,
      Instances data,
      String trainPath,
      Instances test,
      String testPath,
      List<String> externalTestPath,
      String pathToSave,
      int classAct,
      AbstractClassifier cls,
      ClassificationOptimizationParam oP,
      List<ClassificationModelInfo> listClassModelInfo)
      throws ModelingException {
    super(data, trainPath, test, testPath, pathToSave, classAct, mId, externalTestPath, cls);

    listMInfo = listClassModelInfo;
    opt = oP;
  }

  public SearchClassificationModel(
      long mId,
      Instances data,
      String trainPath,
      String testPath,
      List<String> externalTestPath,
      String pathToSave,
      int classAct,
      AbstractClassifier cls,
      ClassificationOptimizationParam oP,
      List<ClassificationModelInfo> listClassModelInfo)
      throws ModelingException {
    super(data, trainPath, testPath, pathToSave, classAct, mId, externalTestPath, cls);

    listMInfo = listClassModelInfo;
    opt = oP;
  }

  @Override
  protected double getEvaluation(
      Instances train, Instances internalTest, List<Instances> externalTests)
      throws ModelingException {
    Set<String> mdNames = getMDNames(train);

    double val = 0;

    try {
      AbstractClassifier clasTmp = (AbstractClassifier) AbstractClassifier.makeCopy(classifier);
      clasTmp.buildClassifier(train);
      Classifier[] copies = AbstractClassifier.makeCopies(clasTmp, 2 + externalTestPath.size());

      Evaluation rfEvaTrain = new Evaluation(train);
      rfEvaTrain.crossValidateModel(copies[1], train, 10, new Random(1));

      double trainACC = rfEvaTrain.pctCorrect();
      double trainSen = sensibilidad(rfEvaTrain.confusionMatrix()) * 100;
      double trainSpe = especificidad(rfEvaTrain.confusionMatrix()) * 100;
      double trainMCC = rfEvaTrain.weightedMatthewsCorrelation();

      double testACC;
      double testSen;
      double testSpe;
      double testMCC;
      if (testPath != null) {
        Evaluation rfEvaTest = new Evaluation(train);
        rfEvaTest.evaluateModel(copies[0], internalTest);
        testACC = rfEvaTest.pctCorrect();
        testSen = sensibilidad(rfEvaTest.confusionMatrix()) * 100;
        testSpe = especificidad(rfEvaTest.confusionMatrix()) * 100;
        testMCC = rfEvaTest.weightedMatthewsCorrelation();
      } else {
        testACC = trainACC;
        testSen = trainSen;
        testSpe = trainSpe;
        testMCC = trainMCC;
      }

      if (Double.isNaN(trainMCC) || Double.isNaN(testMCC)) {
        return 0;
      }

      switch (opt) {
        case ACC_TRAIN:
          val = trainACC;
          break;
        case MCC_TRAIN:
          val = trainMCC;
          break;
        case ACC_TEST:
          val = testACC;
          break;
        case MCC_TEST:
          val = testMCC;
          break;
        case ACC_MEAN:
          val = (trainACC + testACC) / 2;
          break;
        case MCC_MEAN:
          val = (trainMCC + testMCC) / 2;
          break;
      }

      String resExternal = evaluateExternals(train, externalTests, copies);

      String res;
      if (internalTest == null) {
        res =
            buildLineInfoModel(
                trainACC, trainSen, trainSpe, trainMCC, 0, 0, 0, 0, resExternal, mdNames, clasTmp);
      } else {
        res =
            buildLineInfoModel(
                trainACC,
                trainSen,
                trainSpe,
                trainMCC,
                testACC,
                testSen,
                testSpe,
                testMCC,
                resExternal,
                mdNames,
                clasTmp);
      }
      addLineModelToFile(res);
    } catch (Exception ex) {
      throw ModelingException.ExceptionType.BUILDING_MODEL_EXCEPTION.get(
          "Problems building and evaluating the model ", ex);
    }
    return val;
  }

  private void addLineModelToFile(String res) throws ModelingException {

    String line = "";
    if (!(new File(pathToSave).exists())) {
      String head;
      if (testPath == null) head = "classifier,id,size,ACC_CV,SEN_CV,SEP_CV,MCC_CV";
      else
        head = "classifier,id,size,ACC_CV,SEN_CV,SEP_CV,MCC_CV,ACC_Tune,SEN_Tune,SPE_Tune,MCC_Tune";

      String extHead = "";
      if (!externalTestPath.isEmpty()) {
        for (String nameExt : externalTestPath) {
          String singleName = new File(nameExt).getName();
          String headPart =
              String.format(
                  "ACC_%s,SEN_%s,SPE_%s,MCC_%s", singleName, singleName, singleName, singleName);
          if (extHead.isEmpty()) {
            extHead = headPart;
          } else {
            extHead = String.format(FORMAT_HEAD, extHead, headPart);
          }
        }
      }
      line =
          extHead.isEmpty()
              ? String.format("%s,desc\n", head)
              : String.format("%s,%s,desc\n", head, extHead);
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
      double trainACC,
      double trainSen,
      double trainSpe,
      double trainMCC,
      double testACC,
      double testSen,
      double testSpe,
      double testMCC,
      String resExternal,
      Set<String> mdNames,
      AbstractClassifier clasTmp)
      throws ModelingException {
    String res;
    String clasName = getClassifierName(clasTmp);

    res =
        String.format(
            "%s,%d,%d,%.5f,%.5f,%.5f,%.5f",
            clasName, modelId, mdNames.size(), trainACC, trainSen, trainSpe, trainMCC);

    if (testPath != null)
      res = String.format("%s,%.5f,%.5f,%.5f,%.5f", res, testACC, testSen, testSpe, testMCC);

    if (!resExternal.isEmpty()) {
      res = String.format(FORMAT_HEAD, res, resExternal);
    }

    res = String.format(FORMAT_HEAD, res, mdNames.toString().replace(",", " "));

    listMInfo.add(
        new ClassificationModelInfo(
            classifier,
            mdNames.toArray(new String[0]),
            "Ext_" + clasName,
            "",
            mdNames.size(),
            ++modelId,
            trainACC,
            trainSen,
            trainSpe,
            trainMCC,
            testACC,
            testSen,
            testSpe,
            testMCC));
    return res;
  }

  private String evaluateExternals(
      Instances instTrain, List<Instances> extInst, Classifier[] copies) throws ModelingException {
    String resExternal = "";
    int t = 0;
    for (Instances ext : extInst) {
      Evaluation rfEvaExt;
      try {
        rfEvaExt = new Evaluation(instTrain);
        rfEvaExt.evaluateModel(copies[t + 2], ext);
      } catch (Exception e) {
        throw ModelingException.ExceptionType.BUILDING_MODEL_EXCEPTION.get(
            "Problems evaluating model on external sets", e);
      }

      double extACC = rfEvaExt.pctCorrect();
      double extSen = sensibilidad(rfEvaExt.confusionMatrix()) * 100;
      double extSpe = especificidad(rfEvaExt.confusionMatrix()) * 100;
      double extMCC = rfEvaExt.weightedMatthewsCorrelation();
      if (resExternal.isEmpty()) {
        resExternal = String.format("%.5f,%.5f,%.5f,%.5f", extACC, extSen, extSpe, extMCC);
      } else {
        resExternal =
            String.format("%s,%.5f,%.5f,%.5f,%.5f", resExternal, extACC, extSen, extSpe, extMCC);
      }
    }
    return resExternal;
  }

  public double especificidad(double[][] confusionMatrix) {
    double tn = confusionMatrix[0][0];
    double fp = confusionMatrix[0][1];
    return tn / (tn + fp);
  }

  public double sensibilidad(double[][] confusionMatrix) {
    double tp = confusionMatrix[1][1];
    double fn = confusionMatrix[1][0];
    return tp / (tp + fn);
  }
}
