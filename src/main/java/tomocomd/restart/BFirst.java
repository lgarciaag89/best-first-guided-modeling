package tomocomd.restart;

import java.util.*;
import weka.attributeSelection.*;
import weka.core.Instances;

public class BFirst extends BestFirst {

  /** for serialization */
  private static final long serialVersionUID = 7841338689536821867L;

  private BitSet mStartBitSet;
  private boolean isCompleted;

  /** Constructor */
  public BFirst() {
    super();
    isCompleted = false;
  }

  @Override
  public int[] search(ASEvaluation ASEval, Instances data) throws Exception {
    if (isCompleted) {
      return attributeList(mStartBitSet);
    }
    m_totalEvals = 0;
    if (!(ASEval instanceof SubsetEvaluator)) {
      throw new Exception(ASEval.getClass().getName() + " is not a " + "Subset evaluator!");
    }

    if (ASEval instanceof UnsupervisedSubsetEvaluator) {
      m_hasClass = false;
    } else {
      m_hasClass = true;
      m_classIndex = data.classIndex();
    }

    SubsetEvaluator ASEvaluator = (SubsetEvaluator) ASEval;
    m_numAttribs = data.numAttributes();
    int i, j;
    int best_size = 0;
    int size;
    int done;
    int sd = m_searchDirection;
    BitSet best_group, temp_group;
    int stale;
    double best_merit;
    double merit;
    boolean z;
    boolean added;
    Link2 tl;
    Hashtable<String, Double> lookup = new Hashtable<>(m_cacheSize * m_numAttribs);
    int insertCount = 0;
    LinkedList2 bfList = new LinkedList2(m_maxStale);
    stale = 0;
    best_group = new BitSet(m_numAttribs);

    m_startRange.setUpper(m_numAttribs - 1);
    if (!(getStartSet().isEmpty())) {
      m_starting = m_startRange.getSelection();
    }
    // If a starting subset has been supplied, then initialise the bitset
    if (m_starting != null) {
      for (i = 0; i < m_starting.length; i++) {
        if ((m_starting[i]) != m_classIndex) {
          best_group.set(m_starting[i]);
        }
      }

      best_size = m_starting.length;
      m_totalEvals++;
    } else {
      if (m_searchDirection == SELECTION_BACKWARD) {
        setStartSet("1-last");
        m_starting = new int[m_numAttribs];

        // init initial subset to all attributes
        for (i = 0, j = 0; i < m_numAttribs; i++) {
          if (i != m_classIndex) {
            best_group.set(i);
            m_starting[j++] = i;
          }
        }

        best_size = m_numAttribs - 1;
        m_totalEvals++;
      }
    }

    if (Objects.nonNull(mStartBitSet)) {
      if (!(mStartBitSet.isEmpty())) {
        best_group = (BitSet) mStartBitSet.clone();
      }
    }

    // evaluate the initial subset
    best_merit = ASEvaluator.evaluateSubset(best_group);
    // add the initial group to the list and the hash table
    Object[] best = new Object[1];
    best[0] = best_group.clone();
    bfList.addToList(best, best_merit);
    BitSet tt = (BitSet) best_group.clone();
    String hashC = tt.toString();
    lookup.put(hashC, best_merit);

    while (stale < m_maxStale) {
      added = false;

      if (m_searchDirection == SELECTION_BIDIRECTIONAL) {
        // bi-directional search
        done = 2;
        sd = SELECTION_FORWARD;
      } else {
        done = 1;
      }

      // finished search?
      if (bfList.isEmpty()) {
        stale = m_maxStale;
        break;
      }

      // copy the attribute set at the head of the list
      tl = bfList.getLinkAt(0);
      temp_group = (BitSet) (tl.getData()[0]);
      temp_group = (BitSet) temp_group.clone();
      // remove the head of the list
      bfList.removeLinkAt(0);
      // count the number of bits set (attributes)
      int kk;

      for (kk = 0, size = 0; kk < m_numAttribs; kk++) {
        if (temp_group.get(kk)) {
          size++;
        }
      }

      do {
        for (i = 0; i < m_numAttribs; i++) {
          if (sd == SELECTION_FORWARD) {
            z = ((i != m_classIndex) && (!temp_group.get(i)));
          } else {
            z = ((i != m_classIndex) && (temp_group.get(i)));
          }

          if (z) {
            // set the bit (attribute to add/delete)
            if (sd == SELECTION_FORWARD) {
              temp_group.set(i);
              size++;
            } else {
              temp_group.clear(i);
              size--;
            }

            /*
             * if this subset has been seen before, then it is already in the
             * list (or has been fully expanded)
             */
            tt = (BitSet) temp_group.clone();
            hashC = tt.toString();

            if (!lookup.containsKey(hashC)) {
              merit = ASEvaluator.evaluateSubset(temp_group);
              m_totalEvals++;

              // insert this one in the hashtable
              if (insertCount > m_cacheSize * m_numAttribs) {
                lookup = new Hashtable<>(m_cacheSize * m_numAttribs);
                insertCount = 0;
              }
              hashC = tt.toString();
              lookup.put(hashC, merit);
              insertCount++;
            } else {
              merit = lookup.get(hashC);
            }

            // insert this one in the list
            Object[] add = new Object[1];
            add[0] = tt.clone();
            bfList.addToList(add, merit);

            if (m_debug) {
              System.out.print("Group: ");
              printGroup(tt, m_numAttribs);
              System.out.println("Merit: " + merit);
            }

            // is this better than the best?
            if (sd == SELECTION_FORWARD) {
              z = ((merit - best_merit) > 0.00001);
            } else {
              if (merit == best_merit) {
                z = (size < best_size);
              } else {
                z = (merit > best_merit);
              }
            }

            if (z) {
              added = true;
              stale = 0;
              best_merit = merit;
              best_size = size;
              best_group = (BitSet) (temp_group.clone());
              mStartBitSet = (BitSet) (temp_group.clone());
            }

            // unset this addition(deletion)
            if (sd == SELECTION_FORWARD) {
              temp_group.clear(i);
              size--;
            } else {
              temp_group.set(i);
              size++;
            }
          }
        }

        if (done == 2) {
          sd = SELECTION_BACKWARD;
        }

        done--;
      } while (done > 0);

      /*
       * if we haven't added a new attribute subset then full expansion of this
       * node hasen't resulted in anything better
       */
      if (!added) {
        stale++;
      }
    }

    m_bestMerit = best_merit;
    isCompleted = true;
    return attributeList(best_group);
  }
}
