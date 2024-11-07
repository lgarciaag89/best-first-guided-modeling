package tomocomd.searchmodels.v3.performancetracker.classification;

import tomocomd.ModelingException;
import tomocomd.searchmodels.v3.performancetracker.metricvalues.ClassificationMetricValues;
import tomocomd.searchmodels.v3.utils.MetricType;

public class MccMeanModelPerformanceTracker extends AClassificationModelPerformanceTracker {
  @Override
  public MetricType getMetricType() {
    return MetricType.MCC_MEAN;
  }

  @Override
  public double computeValueToCompare() throws ModelingException {
    return (((ClassificationMetricValues) trainValues).getMcc()
            + ((ClassificationMetricValues) testValues).getMcc())
        / 2;
  }
}
