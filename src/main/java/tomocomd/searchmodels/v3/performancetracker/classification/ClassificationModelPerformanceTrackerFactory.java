package tomocomd.searchmodels.v3.performancetracker.classification;

import tomocomd.searchmodels.v3.performancetracker.AModelPerformanceTracker;
import tomocomd.searchmodels.v3.utils.MetricType;

public class ClassificationModelPerformanceTrackerFactory {

  public static AModelPerformanceTracker getClassificationModelPerformanceTracker(MetricType type) {
    switch (type) {
      case ACC_TRAIN:
        return new AccTrainModelPerformanceTracker();
      case ACC_TEST:
        return new AccTestModelPerformanceTracker();
      case ACC_MEAN:
        return new AccMeanModelPerformanceTracker();
      case MCC_TRAIN:
        return new MccTrainModelPerformanceTracker();
      case MCC_TEST:
        return new MccTestModelPerformanceTracker();
      case MCC_MEAN:
        return new MccMeanModelPerformanceTracker();
      default:
        throw new IllegalArgumentException("Unknown metric type: " + type);
    }
  }
}
