package tomocomd.searchmodels.v3.performancetracker.regression;

import static org.junit.jupiter.api.Assertions.*;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;
import tomocomd.BuildClassifierList;
import tomocomd.ClassifierNameEnum;
import tomocomd.searchmodels.v3.utils.MetricType;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

class MaeMeanModelPerformanceTrackerTest {
  MaeMeanModelPerformanceTracker tracker = new MaeMeanModelPerformanceTracker();

  @Test
  void testIsClassification() {
    assertFalse(tracker.isClassification());
  }

  @Test
  void testGetMetricType() {
    assertEquals(MetricType.MAE_MEAN, tracker.getMetricType());
  }

  @Test
  void buildAndGetClassifierName() {
    List<ClassifierNameEnum> classifierNameList =
        BuildClassifierList.getClassifierNameList(new String[] {"all"}, false);

    classifierNameList.forEach(
        classifierName -> {
          Classifier[] classList = tracker.buildAndGetClassifiers(classifierName, getTrain(), 2);
          assertEquals(2, classList.length);
          assertNotNull(classList[0]);
        });
  }

  Instances getTrain() {
    ArrayList<Attribute> attributes = new ArrayList<>();
    attributes.add(new Attribute("attr1"));
    attributes.add(new Attribute("attr2"));
    attributes.add(new Attribute("class"));

    Instances train = new Instances("Dataset", attributes, 4);
    train.setClassIndex(2);

    double[] instanceValue1 = {1.0, 2.0, 0.0};
    double[] instanceValue2 = {3.0, 4.0, 1.0};
    double[] instanceValue3 = {5.0, 6.0, 0.0};
    double[] instanceValue4 = {7.0, 8.0, 1.0};

    train.add(new DenseInstance(1.0, instanceValue1));
    train.add(new DenseInstance(1.0, instanceValue2));
    train.add(new DenseInstance(1.0, instanceValue3));
    train.add(new DenseInstance(1.0, instanceValue4));

    return train;
  }
}
