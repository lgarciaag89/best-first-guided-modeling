package tomocomd.restart;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import tomocomd.searchmodels.v3.InitSearchModel;

public class StatusManager {

  public static void saveStatus(InitSearchModel initSearchModel, String saveFile) {
    String tmpFile = saveFile + ".tmp";
    Path tmpPath = Paths.get(tmpFile);
    Path finalPath = Paths.get(saveFile);

    try (ObjectOutputStream oos = new ObjectOutputStream(Files.newOutputStream(tmpPath))) {
      oos.writeObject(initSearchModel);
    } catch (IOException e) {
      System.err.printf("Failed to write temp model to: %s%n", tmpFile);
      e.printStackTrace();
      System.exit(-1);
    }

    try {
      Files.copy(tmpPath, finalPath, StandardCopyOption.REPLACE_EXISTING);
      Files.delete(tmpPath);
      System.out.printf("Auto-saved model to: %s%n", saveFile);
    } catch (IOException e) {
      System.err.printf("Failed to finalize auto-save to: %s%n", saveFile);
      e.printStackTrace();
      System.exit(-1);
    }
  }

  public static InitSearchModel loadStatus(File statusFile)
      throws IOException, ClassNotFoundException {

    try (ObjectInputStream ois = new ObjectInputStream(Files.newInputStream(statusFile.toPath()))) {
      System.out.println("Status loading from: " + statusFile.getAbsolutePath());
      return (InitSearchModel) ois.readObject();
    }
  }
}
