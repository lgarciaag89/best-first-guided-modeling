package tomocomd.filters;

import java.io.File;
import java.util.Objects;
import tomocomd.CSVManage;
import tomocomd.utils.Constants;
import tomocomd.utils.ModelingException;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Reorder;

public class ReorderData {

  public static void reorder(File train, File tune, File ext, String act) throws ModelingException {
    Instances tempTrainData = CSVManage.loadCSV(train.getAbsolutePath());

    int posAct = searchActPos(tempTrainData, act);

    StringBuilder sb = new StringBuilder();
    if (posAct == -1 || posAct == tempTrainData.numAttributes() - 1) {
      sb.append("last-first");
    } else if (posAct == 0) {
      sb.append("1,last-2");
    } else {
      sb.append(posAct + 1)
          .append(",last-")
          .append(posAct + 2)
          .append(",")
          .append(posAct)
          .append("-first");
    }

    CSVManage.saveDescriptorMResult(
        executeRorder(tempTrainData, sb.toString()),
        train.getAbsolutePath() + Constants.REVERSE_MARK);
    if (Objects.nonNull(tune)) {
      CSVManage.saveDescriptorMResult(
          executeRorder(CSVManage.loadCSV(tune.getAbsolutePath()), sb.toString()),
          tune.getAbsolutePath() + Constants.REVERSE_MARK);
    }
    if (Objects.nonNull(ext)) {
      File externalFilterFolder = new File(ext.getAbsolutePath() + Constants.REVERSE_MARK_FOLDER);
      externalFilterFolder.mkdir();
      for (File extFile : Objects.requireNonNull(ext.listFiles())) {
        CSVManage.saveDescriptorMResult(
            executeRorder(CSVManage.loadCSV(extFile.getAbsolutePath()), sb.toString()),
            new File(externalFilterFolder, extFile.getName() + Constants.REVERSE_MARK)
                .getAbsolutePath());
      }
    }
  }

  public static Instances executeRorder(Instances data, String positions) throws ModelingException {
    Instances filteredData = new Instances(data);
    Reorder reorder = new Reorder();

    try {
      reorder.setAttributeIndices(positions);
      reorder.setInputFormat(filteredData);
      return Filter.useFilter(filteredData, reorder);
    } catch (Exception e) {
      throw ModelingException.ExceptionType.FILTERING_EXCEPTION.get("Error applying filters", e);
    }
  }

  private static int searchActPos(Instances data, String act) {
    for (int i = 0; i < data.numAttributes(); i++)
      if (data.attribute(i).name().equals(act)) return i;
    return -1;
  }
}
