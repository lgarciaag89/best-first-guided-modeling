package tomocomd.restart;

import java.io.*;
import java.util.concurrent.*;
import tomocomd.searchmodels.v3.InitSearchModel;

public class ModelAutoSaver {

  private static final Integer DELAY_INTERVAL_SECONDS = 10;
  private static final Integer PERIOD_INTERVAL_SECONDS = 30;

  private final ScheduledExecutorService executorService;
  private final File saveFile;
  private final InitSearchModel initSearchModel;

  public ModelAutoSaver(InitSearchModel initSearchModel, File saveFile) {
    this.saveFile = saveFile;
    this.initSearchModel = initSearchModel;
    this.executorService = Executors.newSingleThreadScheduledExecutor();
  }

  public void startAutoSave() {
    executorService.scheduleAtFixedRate(
        this::saveObject, DELAY_INTERVAL_SECONDS, PERIOD_INTERVAL_SECONDS, TimeUnit.SECONDS);
  }

  public void stopAutoSave() {
    saveObject();
    executorService.shutdownNow();
  }

  private void saveObject() {
    StatusManager.saveStatus(initSearchModel, saveFile.getAbsolutePath());
    System.out.println("Saved status");
  }
}
