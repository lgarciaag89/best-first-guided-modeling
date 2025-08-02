package tomocomd.searchmodels.v3.performancetracker.regression;

import tomocomd.searchmodels.v3.performancetracker.metricvalues.RegressionMetricValues;
import tomocomd.searchmodels.v3.utils.MetricType;
import tomocomd.utils.ModelingException;

public class MaeTestModelPerformanceTracker extends ARegressionModelPerformanceTracker {
  @Override
  public MetricType getMetricType() {
    return MetricType.MAE_EXT;
  }

  @Override
  public double computeValueToCompare() throws ModelingException {
    return ((RegressionMetricValues) testValues).getMae();
  }
}
