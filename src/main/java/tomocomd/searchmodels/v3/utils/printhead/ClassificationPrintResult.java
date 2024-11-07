package tomocomd.searchmodels.v3.utils.printhead;

import java.util.List;
import java.util.stream.Collectors;

public class ClassificationPrintResult extends APrintResult {

  @Override
  public String generateHead(boolean hasTune, List<String> externalTestPath) {
    String head =
        hasTune
            ? "classifier,id,size,ACC_CV,SEN_CV,SPE_CV,MCC_CV,ACC_Tune,SEN_Tune,SPE_Tune,MCC_Tune"
            : "classifier,id,size,ACC_CV,SEN_CV,SPE_CV,MCC_CV";
    String externalHead =
        externalTestPath.stream()
            .map(name -> String.format("ACC_%s,SEN_%s,SPE_%s,MCC_%s", name, name, name, name))
            .collect(Collectors.joining(","));
    return head + (externalHead.isEmpty() ? "" : "," + externalHead) + ",Attributes";
  }
}
