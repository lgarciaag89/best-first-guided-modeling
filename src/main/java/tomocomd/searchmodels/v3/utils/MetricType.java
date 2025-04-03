package tomocomd.searchmodels.v3.utils;

public enum MetricType {
  ACC_TRAIN(ProblemType.CLASSIFICATION),
  ACC_TEST(ProblemType.CLASSIFICATION),
  ACC_MEAN(ProblemType.CLASSIFICATION),
  MCC_TRAIN(ProblemType.CLASSIFICATION),
  MCC_TEST(ProblemType.CLASSIFICATION),
  MCC_MEAN(ProblemType.CLASSIFICATION),
  Q2_TRAIN(ProblemType.REGRESSION),
  Q2_EXT(ProblemType.REGRESSION),
  MAE_TRAIN(ProblemType.REGRESSION),
  MAE_EXT(ProblemType.REGRESSION),
  RMSE_TRAIN(ProblemType.REGRESSION),
  RMSE_EXT(ProblemType.REGRESSION),
  Q2_MEAN(ProblemType.REGRESSION),
  MAE_MEAN(ProblemType.REGRESSION),
  RMSE_MEAN(ProblemType.REGRESSION);

  private final ProblemType problemType;

  MetricType(ProblemType problemType) {
    this.problemType = problemType;
  }

  public ProblemType getProblemType() {
    return problemType;
  }

  public enum ProblemType {
    REGRESSION,
    CLASSIFICATION,
    REGRESSION_CLASSIFICATION
  }
}
