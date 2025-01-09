package tomocomd.utils;

import java.io.InputStream;
import java.util.jar.Attributes;
import java.util.jar.Manifest;

public class VersionUtil {
  public static String getVersionInfo() {
    try (InputStream manifestStream =
        VersionUtil.class.getClassLoader().getResourceAsStream("META-INF/MANIFEST.MF")) {

      if (manifestStream == null) {
        return "Unknown version (manifest not found)";
      }

      Manifest manifest = new Manifest(manifestStream);
      Attributes attr = manifest.getMainAttributes();
      String version = attr.getValue("Implementation-Version");
      String javaVersion = attr.getValue("Build-Java-Version");

      return String.format(
          "best-first-guided-modeling Version: %s\nJava Version: %s",
          version != null ? version : "Unknown", javaVersion != null ? javaVersion : "Unknown");
    } catch (Exception e) {
      return "Unknown version (error reading manifest)";
    }
  }

  public static void main(String[] args) {
    System.out.println(getVersionInfo());
  }
}
