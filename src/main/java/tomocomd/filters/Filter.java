package tomocomd.filters;

import tomocomd.utils.ModelingException;
import weka.core.Instances;

public interface Filter {

  Boolean passFilter(Instances data, Integer attributeIdx, Double threshold)
      throws ModelingException;

  FilterType getType();
}
