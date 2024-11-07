package tomocomd.searchmodels.deprecated.v1;

import java.util.*;
import tomocomd.ModelingException;
import tomocomd.searchmodels.deprecated.ABaseSearchModel;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

public abstract class ASearchModelByOneClasifier extends ABaseSearchModel {

  protected final AbstractClassifier classifier;

  protected ASearchModelByOneClasifier(
      String trainPath,
      String testPath,
      String pathToSave,
      int classAct,
      long modelId,
      List<String> externalTestPath,
      AbstractClassifier classifier)
      throws ModelingException {
    super(trainPath, testPath, pathToSave, classAct, modelId, externalTestPath);
    try {
      this.classifier = (AbstractClassifier) AbstractClassifier.makeCopy(classifier);
    } catch (Exception e) {
      throw ModelingException.ExceptionType.CLASSIFIER_LOAD_EXCEPTION.get(ERROR_MSG, e);
    }
  }

  protected ASearchModelByOneClasifier(
      Instances trainTotal,
      String trainPath,
      String testPath,
      String pathToSave,
      int classAct,
      long modelId,
      List<String> externalTestPath,
      AbstractClassifier classifier)
      throws ModelingException {
    super(trainTotal, trainPath, testPath, pathToSave, classAct, modelId, externalTestPath);
    try {
      this.classifier = (AbstractClassifier) AbstractClassifier.makeCopy(classifier);
    } catch (Exception e) {
      throw ModelingException.ExceptionType.CLASSIFIER_LOAD_EXCEPTION.get(ERROR_MSG, e);
    }
  }

  protected ASearchModelByOneClasifier(
      Instances trainTotal,
      String trainPath,
      Instances testTotal,
      String testPath,
      String pathToSave,
      int classAct,
      long modelId,
      List<String> externalTestPath,
      AbstractClassifier classifier)
      throws ModelingException {
    super(
        trainTotal,
        trainPath,
        testTotal,
        testPath,
        pathToSave,
        classAct,
        modelId,
        externalTestPath);
    try {
      this.classifier = (AbstractClassifier) AbstractClassifier.makeCopy(classifier);
    } catch (Exception e) {
      throw ModelingException.ExceptionType.CLASSIFIER_LOAD_EXCEPTION.get(ERROR_MSG, e);
    }
  }
}
