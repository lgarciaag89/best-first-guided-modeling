package tomocomd.searchmodels.v3.performancetracker.regression;

import tomocomd.searchmodels.v3.performancetracker.AModelPerformanceTracker;
import tomocomd.searchmodels.v3.utils.MetricType;

public class RegressionModelPerformanceTrackerFactory {

  public static AModelPerformanceTracker getRegressionModelPerformanceTracker(
      MetricType metricType) {

    switch (metricType) {
      case Q2_TRAIN:
        return new R2TrainModelPerformanceTracker();
      case Q2_EXT:
        return new R2TestModelPerformanceTracker();
      case Q2_MEAN:
        return new R2MeanModelPerformanceTracker();
      case MAE_TRAIN:
        return new MaeTrainModelPerformanceTracker();
      case MAE_EXT:
        return new MaeTestModelPerformanceTracker();
      case MAE_MEAN:
        return new MaeMeanModelPerformanceTracker();
      case RMSE_TRAIN:
        return new RmseTrainModelPerformanceTracker();
      case RMSE_EXT:
        return new RmseTestModelPerformanceTracker();
      case RMSE_MEAN:
        return new RmseMeanModelPerformanceTracker();
      default:
        throw new IllegalArgumentException("Unknown metric type: " + metricType);
    }
  }
}
