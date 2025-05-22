package tomocomd.restart;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import tomocomd.searchmodels.v3.InitSearchModel;

public class StatusManager {

  public static void saveStatus(InitSearchModel initSearchModel, String saveFile) {
    String tmpFile = saveFile + ".tmp";
    Path path = Paths.get(tmpFile);
    try (ObjectOutputStream oos = new ObjectOutputStream(Files.newOutputStream(path))) {
      oos.writeObject(initSearchModel);
      Files.move(
          path,
          Paths.get(saveFile),
          java.nio.file.StandardCopyOption.REPLACE_EXISTING,
          java.nio.file.StandardCopyOption.ATOMIC_MOVE);
      System.out.printf("Auto-saved model to: %s%n", saveFile);
    } catch (IOException e) {
      System.err.printf("Failed to auto-save model to: %s%n", saveFile);
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
