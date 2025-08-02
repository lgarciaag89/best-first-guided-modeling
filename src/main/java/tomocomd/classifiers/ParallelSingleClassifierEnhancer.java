package tomocomd.classifiers;

/*
 * This is a copy of weka ParallelIteratedSingleClassifierEnhancer for
 * Abstract utility class for handling settings common to meta classifiers that
 * build an ensemble in parallel from a single base learner.
 */

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import tomocomd.utils.ModelingException;
import weka.classifiers.Classifier;
import weka.classifiers.IteratedSingleClassifierEnhancer;
import weka.core.Instances;

public abstract class ParallelSingleClassifierEnhancer extends IteratedSingleClassifierEnhancer {

  private static final Logger logger = LogManager.getLogger(ParallelSingleClassifierEnhancer.class);

  /** For serialization */
  private static final long serialVersionUID = 202406131234567890L;

  /** The number of threads to have executing at any one time */
  protected final int mNumExecutionSlots;

  protected final String classifierName;

  protected ParallelSingleClassifierEnhancer(int mNumExecutionSlots, String classifierName) {
    super();
    this.mNumExecutionSlots = mNumExecutionSlots;
    this.classifierName = classifierName;
  }

  public String getClassifierName() {
    return classifierName;
  }

  /**
   * Stump method for building the classifiers
   *
   * @param data the training data to be used for generating the ensemble
   * @throws Exception if the classifier could not be built successfully
   */
  @Override
  public void buildClassifier(Instances data) throws Exception {
    if (mNumExecutionSlots < 0) {
      throw ModelingException.ExceptionType.MULTITHREADING_EXCEPTION.get(
          "Number of execution slots needs to be >= 0!");
    }
    super.buildClassifier(data);
  }

  /*
   * Start the pool of execution threads
   *
   * Does the actual construction of the ensemble
   *
   * @throws Exception if something goes wrong during the training process
   */
  protected void buildClassifiers() throws Exception {

    if (mNumExecutionSlots == 1) {
      runSequential();
      return;
    }

    int numCores =
        (mNumExecutionSlots == 0) ? Runtime.getRuntime().availableProcessors() : mNumExecutionSlots;
    ExecutorService executorPool = Executors.newFixedThreadPool(numCores);

    final CountDownLatch doneSignal = new CountDownLatch(m_Classifiers.length);
    final AtomicInteger numFailed = new AtomicInteger();

    for (int i = 0; i < m_Classifiers.length; i++) {

      final Classifier currentClassifier = m_Classifiers[i];
      // MultiClassClassifier may produce occasional NULL classifiers ...
      if (currentClassifier == null) continue;
      final int iteration = i;

      logger.info(
          "Training instance classifier ({}) for {} meta classifier... ", i + 1, classifierName);

      Runnable newTask =
          () -> {
            try {
              currentClassifier.buildClassifier(getTrainingSet(iteration));
            } catch (Exception ex) {
              numFailed.incrementAndGet();
              logger.error(
                  "Instance classifier ({}) for {} meta classifier failed: ",
                  (iteration + 1),
                  classifierName,
                  ex);
            } finally {
              doneSignal.countDown();
            }
          };
      // launch this task
      executorPool.submit(newTask);
    }
    // wait for all tasks to finish, then shutdown pool
    doneSignal.await();
    executorPool.shutdownNow();
    if (m_Debug && numFailed.intValue() > 0) {
      logger.error("Problem building classifiers {} - some iterations failed.", classifierName);
    }
  }

  private void runSequential() throws Exception {
    // simple single-threaded execution
    for (int i = 0; i < m_Classifiers.length; i++) {
      m_Classifiers[i].buildClassifier(getTrainingSet(i));
    }
  }

  /**
   * Gets a training set for a particular iteration. Implementations need to be careful with thread
   * safety and should probably be synchronized to be on the safe side.
   *
   * @param iteration the number of the iteration for the requested training set
   * @return the training set for the supplied iteration number
   * @throws ModelingException if something goes wrong.
   */
  protected abstract Instances getTrainingSet(int iteration) throws ModelingException;
}
