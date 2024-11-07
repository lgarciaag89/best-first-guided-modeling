package tomocomd.searchmodels.v3.performancetracker.regression;

import tomocomd.ModelingException;
import tomocomd.searchmodels.v3.performancetracker.metricvalues.RegressionMetricValues;
import tomocomd.searchmodels.v3.utils.MetricType;

public class RmseMeanModelPerformanceTracker extends ARegressionModelPerformanceTracker {
  @Override
  public MetricType getMetricType() {
    return MetricType.RMSE_MEAN;
  }

  @Override
  public double computeValueToCompare() throws ModelingException {
    return (((RegressionMetricValues) trainValues).getRmse()
            + ((RegressionMetricValues) testValues).getRmse())
        / 2;
  }
}
