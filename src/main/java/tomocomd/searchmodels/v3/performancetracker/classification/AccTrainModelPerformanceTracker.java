package tomocomd.searchmodels.v3.performancetracker.classification;

import tomocomd.searchmodels.v3.performancetracker.metricvalues.ClassificationMetricValues;
import tomocomd.searchmodels.v3.utils.MetricType;
import tomocomd.utils.ModelingException;

public class AccTrainModelPerformanceTracker extends AClassificationModelPerformanceTracker {
  @Override
  public MetricType getMetricType() {
    return MetricType.ACC_TRAIN;
  }

  @Override
  public double computeValueToCompare() throws ModelingException {
    return ((ClassificationMetricValues) trainValues).getAcc();
  }
}
