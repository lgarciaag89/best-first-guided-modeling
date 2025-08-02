package tomocomd.classifiers;

import weka.core.Randomizable;

/**
 * Abstract utility class based on weka RandomizableParallelIteratedSingleClassifierEnhancer for
 * handling settings common to randomizable meta classifiers that build an ensemble in parallel from
 * a single base learner.
 */
public abstract class RandomizableParallelEnhancer extends ParallelSingleClassifierEnhancer
    implements Randomizable {

  /** For serialization */
  private static final long serialVersionUID = 987654321012346789L;

  /** The random number seed. */
  protected int mSeed;

  /** Default constructor. */
  protected RandomizableParallelEnhancer() {
    this(1, 1, "Unknown");
  }

  protected RandomizableParallelEnhancer(int mSeed, int mNumExecutionSlots, String name) {
    super(mNumExecutionSlots, name);
    this.mSeed = mSeed;
  }

  /**
   * Gets the seed for the random number generations
   *
   * @return the seed for the random number generation
   */
  public int getSeed() {

    return mSeed;
  }

  /**
   * Sets the seed for the random number generations
   *
   * @param seed the seed for the random number generation
   */
  public void setSeed(int seed) {
    this.mSeed = seed;
  }
}
