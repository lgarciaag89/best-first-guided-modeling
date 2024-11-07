package tomocomd.searchmodels.v3.utils.printhead;

import java.util.List;
import java.util.stream.Collectors;

public class RegressionPrintResult extends APrintResult {

  @Override
  public String generateHead(boolean hasTune, List<String> externalTestPath) {
    String head =
        hasTune
            ? "classifier,id,size,Q2_CV,MAE_CV, RMSE_CV,Q2_Tune,MAE_Tune, RMSE_TUNE"
            : "classifier,id,size,Q2_CV,MAE_CV, RMSE_CV";
    String externalHead =
        externalTestPath.stream()
            .map(name -> String.format("Q2_%s,MAE_%s,RMSE_%s", name, name, name))
            .collect(Collectors.joining(","));
    return head + (externalHead.isEmpty() ? "" : "," + externalHead) + ",Attributes";
  }
}
