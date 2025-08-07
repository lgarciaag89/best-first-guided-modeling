package tomocomd;

import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import org.junit.jupiter.api.Test;
import tomocomd.classifiers.BuildClassifierList;
import tomocomd.classifiers.ClassifierNameEnum;
import tomocomd.searchmodels.v3.utils.MetricType;

class BuildClassifierListTest {

  @Test
  void getClassifierNameListClass() {
    String[] models = {"KNN", "RANDOMFOREST", "ADABOOST", "ADDITIVEREGRESSION-RF"};

    List<ClassifierNameEnum> classifierNameList =
        BuildClassifierList.getClassifierNameList(models, true);
    assertEquals(3, classifierNameList.size());
    assertTrue(classifierNameList.contains(ClassifierNameEnum.KNN));
    assertTrue(classifierNameList.contains(ClassifierNameEnum.RANDOMFOREST));
    assertTrue(classifierNameList.contains(ClassifierNameEnum.ADABOOST));
  }

  @Test
  void getClassifierNameListReg() {
    String[] models = {"KNN", "RANDOMFOREST", "ADABOOST", "ADDITIVEREGRESSION-rf"};

    List<ClassifierNameEnum> classifierNameList =
        BuildClassifierList.getClassifierNameList(models, false);
    assertEquals(3, classifierNameList.size());
    assertTrue(classifierNameList.contains(ClassifierNameEnum.KNN));
    assertTrue(classifierNameList.contains(ClassifierNameEnum.RANDOMFOREST));
    assertTrue(classifierNameList.contains(ClassifierNameEnum.ADDITIVEREGRESSION_RF));
  }

  @Test
  void getClassifierNameListClassALL() {
    String[] models = {"all"};

    List<ClassifierNameEnum> classifierNameList =
        BuildClassifierList.getClassifierNameList(models, true);
    assertEquals(11, classifierNameList.size());

    classifierNameList.forEach(
        classifierNameEnum ->
            assertTrue(
                classifierNameEnum.getProblemType() == MetricType.ProblemType.CLASSIFICATION
                    || classifierNameEnum.getProblemType()
                        == MetricType.ProblemType.REGRESSION_CLASSIFICATION));
  }

  @Test
  void getClassifierNameListRegALL() {
    String[] models = {"all"};

    List<ClassifierNameEnum> classifierNameList =
        BuildClassifierList.getClassifierNameList(models, false);
    assertEquals(12, classifierNameList.size());

    classifierNameList.forEach(
        classifierNameEnum ->
            assertTrue(
                classifierNameEnum.getProblemType() == MetricType.ProblemType.REGRESSION
                    || classifierNameEnum.getProblemType()
                        == MetricType.ProblemType.REGRESSION_CLASSIFICATION));
  }
}
