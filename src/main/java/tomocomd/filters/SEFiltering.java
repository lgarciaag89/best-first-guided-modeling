package tomocomd.filters;

import tomocomd.utils.ModelingException;
import tomocomd.utils.SEAttribute;
import weka.core.Instances;

public class SEFiltering implements Filter {
  public SEFiltering() {}

  @Override
  public Boolean passFilter(Instances data, Integer attributeIdx, Double threshold)
      throws ModelingException {
    double maxSE = Math.log(data.numInstances());

    double entropy = SEAttribute.computeEntropy(data.attributeToDoubleArray(attributeIdx));
    data.attribute(attributeIdx).setWeight(entropy);
    return entropy > maxSE * threshold;
  }

  @Override
  public FilterType getType() {
    return FilterType.SE;
  }
}
