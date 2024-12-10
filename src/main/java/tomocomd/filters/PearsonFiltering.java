package tomocomd.filters;

import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import tomocomd.ModelingException;
import weka.core.Instances;

public class PearsonFiltering implements Filter {
  public PearsonFiltering() {}

  @Override
  public Boolean passFilter(Instances data, Integer attributeIdx, Double threshold)
      throws ModelingException {
    if (data.classIndex() < 0)
      throw ModelingException.ExceptionType.FILTERING_EXCEPTION.get("Class index not set");
    PearsonsCorrelation pearsonsCorrelation = new PearsonsCorrelation();
    double[] classValues = data.attributeToDoubleArray(data.classIndex());
    double[] attributeValues = data.attributeToDoubleArray(attributeIdx);
    double correlation = pearsonsCorrelation.correlation(classValues, attributeValues);
    return correlation > threshold;
  }

  @Override
  public FilterType getType() {
    return FilterType.R;
  }
}
