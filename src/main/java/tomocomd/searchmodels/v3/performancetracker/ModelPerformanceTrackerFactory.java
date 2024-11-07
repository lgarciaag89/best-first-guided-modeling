package tomocomd.searchmodels.v3.performancetracker;

import tomocomd.searchmodels.v3.performancetracker.classification.ClassificationModelPerformanceTrackerFactory;
import tomocomd.searchmodels.v3.performancetracker.regression.RegressionModelPerformanceTrackerFactory;
import tomocomd.searchmodels.v3.utils.MetricType;

public class ModelPerformanceTrackerFactory {

  public static AModelPerformanceTracker createModelPerformanceTracker(MetricType metricType) {

    switch (metricType.getProblemType()) {
      case CLASSIFICATION:
        return ClassificationModelPerformanceTrackerFactory
            .getClassificationModelPerformanceTracker(metricType);
      case REGRESSION:
        return RegressionModelPerformanceTrackerFactory.getRegressionModelPerformanceTracker(
            metricType);
      default:
        throw new IllegalArgumentException("Unknown problem type: " + metricType.getProblemType());
    }
  }
}
