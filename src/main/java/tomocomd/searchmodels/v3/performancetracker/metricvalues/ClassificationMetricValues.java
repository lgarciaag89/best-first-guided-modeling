package tomocomd.searchmodels.v3.performancetracker.metricvalues;

public class ClassificationMetricValues implements IMetricValue {
  double acc;
  double sen;
  double spe;
  double mcc;

  public ClassificationMetricValues(double acc, double sen, double spe, double mcc) {
    this.acc = acc;
    this.sen = sen;
    this.spe = spe;
    this.mcc = mcc;
  }

  public double getAcc() {
    return acc;
  }

  public double getSen() {
    return sen;
  }

  public double getSpe() {
    return spe;
  }

  public double getMcc() {
    return mcc;
  }
}
