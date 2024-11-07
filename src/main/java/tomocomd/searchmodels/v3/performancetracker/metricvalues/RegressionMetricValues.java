package tomocomd.searchmodels.v3.performancetracker.metricvalues;

public class RegressionMetricValues implements IMetricValue {
  double q2;
  double mae;
  double rmse;

  public RegressionMetricValues(double q2, double mae, double rmse) {
    this.q2 = q2;
    this.mae = mae;
    this.rmse = rmse;
  }

  public double getQ2() {
    return q2;
  }

  public double getMae() {
    return mae;
  }

  public double getRmse() {
    return rmse;
  }
}
