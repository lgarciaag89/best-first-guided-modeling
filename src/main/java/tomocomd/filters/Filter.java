package tomocomd.filters;

import tomocomd.ModelingException;
import weka.core.Instances;

public interface Filter {

  Boolean passFilter(Instances data, Integer attributeIdx, Double threshold)
      throws ModelingException;

  FilterType getType();
}
