package tomocomd.searchmodels.v3.performancetracker.regression;

import tomocomd.ModelingException;
import tomocomd.searchmodels.v3.performancetracker.metricvalues.RegressionMetricValues;
import tomocomd.searchmodels.v3.utils.MetricType;

public class R2MeanModelPerformanceTracker extends ARegressionModelPerformanceTracker {
  @Override
  public MetricType getMetricType() {
    return MetricType.Q2_MEAN;
  }

  @Override
  public double computeValueToCompare() throws ModelingException {
    return (((RegressionMetricValues) trainValues).getQ2()
            + ((RegressionMetricValues) testValues).getQ2())
        / 2;
  }
}
