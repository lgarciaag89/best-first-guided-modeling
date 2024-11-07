package tomocomd.searchmodels.deprecated.v2;

import java.util.*;
import java.util.stream.Collectors;
import tomocomd.ModelingException;
import tomocomd.searchmodels.deprecated.ABaseSearchModel;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

public abstract class ASearchModelBySeveralClassifiers extends ABaseSearchModel {

  protected final List<AbstractClassifier> classifiers;

  protected ASearchModelBySeveralClassifiers(
      String trainPath,
      String testPath,
      String pathToSave,
      int classAct,
      long modelId,
      List<String> externalTestPath,
      List<AbstractClassifier> classifiers)
      throws ModelingException {
    super(trainPath, testPath, pathToSave, classAct, modelId, externalTestPath);
    try {
      this.classifiers = new LinkedList<>(classifiers);
    } catch (Exception e) {
      throw ModelingException.ExceptionType.CLASSIFIER_LOAD_EXCEPTION.get(ERROR_MSG, e);
    }
  }

  protected ASearchModelBySeveralClassifiers(
      Instances trainTotal,
      String trainPath,
      String testPath,
      String pathToSave,
      int classAct,
      long modelId,
      List<String> externalTestPath,
      List<AbstractClassifier> classifiers)
      throws ModelingException {
    super(trainTotal, trainPath, testPath, pathToSave, classAct, modelId, externalTestPath);
    try {
      this.classifiers = new LinkedList<>(classifiers);
    } catch (Exception e) {
      throw ModelingException.ExceptionType.CLASSIFIER_LOAD_EXCEPTION.get(ERROR_MSG, e);
    }
  }

  protected ASearchModelBySeveralClassifiers(
      Instances trainTotal,
      String trainPath,
      Instances testTotal,
      String testPath,
      String pathToSave,
      int classAct,
      long modelId,
      List<String> externalTestPath,
      List<AbstractClassifier> classifiers)
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
      this.classifiers = new LinkedList<>(classifiers);
    } catch (Exception e) {
      throw ModelingException.ExceptionType.CLASSIFIER_LOAD_EXCEPTION.get(ERROR_MSG, e);
    }
  }

  @Override
  protected double getEvaluation(
      Instances train, Instances internalTest, List<Instances> externalTests)
      throws ModelingException {
    Set<String> mdNames = getMDNames(train);

    double val = 0;
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
      val =
          Math.max(
              val,
              evaluateOneClass(
                  localTrain, localInternalTest, localExternalTests, mdNames, classifier));
    }
    return val;
  }

  protected abstract double evaluateOneClass(
      Instances localTrain,
      Instances localInternalTest,
      List<Instances> localExternalTests,
      Set<String> mdNames,
      AbstractClassifier classifier);
}
