package tomocomd.utils;

import tomocomd.ModelingException;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Remove;

public class Removing {

  public static Instances executeRemove(Instances data, int[] pos, boolean invert) {
    Instances filteredData = new Instances(data);
    Remove remove = new Remove();
    remove.setAttributeIndicesArray(pos);
    remove.setInvertSelection(invert);
    try {
      remove.setInputFormat(filteredData);
      return Remove.useFilter(filteredData, remove);
    } catch (Exception e) {
      throw ModelingException.ExceptionType.FILTERING_EXCEPTION.get("Error applying filters", e);
    }
  }
}
