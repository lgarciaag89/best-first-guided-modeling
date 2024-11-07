package tomocomd.searchmodels.v3.utils.printhead;

import tomocomd.searchmodels.v3.utils.MetricType;

public class PrintResultFactory {

  public static APrintResult createPrintResult(MetricType type) {
    switch (type.getProblemType()) {
      case REGRESSION:
        return new RegressionPrintResult();
      case CLASSIFICATION:
        return new ClassificationPrintResult();
      default:
        throw new IllegalArgumentException("Unknown type: " + type);
    }
  }
}
