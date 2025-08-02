package tomocomd.searchmodels.v3.utils.printhead;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.List;
import tomocomd.utils.ModelingException;

public abstract class APrintResult implements Serializable {

  protected abstract String generateHead(boolean hasTune, List<String> externalTestPath);

  public void printLine(String line, String pathToSave) {
    try (FileWriter fw = new FileWriter(pathToSave, true);
        BufferedWriter w = new BufferedWriter(fw)) {
      w.write(line + "\n");
    } catch (IOException ex) {
      throw ModelingException.ExceptionType.CSV_FILE_WRITING_EXCEPTION.get(
          "Problems saving models results on file " + pathToSave, ex);
    }
  }

  public void createHead(boolean hasTune, List<String> externalTestPath, String pathToSave) {
    String head = generateHead(hasTune, externalTestPath);
    printLine(head, pathToSave);
  }
}
