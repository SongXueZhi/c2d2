#! /usr/bin/env python3
# $Id: DD.py,v 1.2 2001/11/05 19:53:33 zeller Exp $
# Enhanced Delta Debugging class
# Copyright (c) 1999, 2000, 2001 Andreas Zeller.


# This module (written in Python) implements the base delta debugging
# algorithms and is at the core of all our experiments.  This should
# easily run on any platform and any Python version since 1.6.
#
# To plug this into your system, all you have to do is to create a
# subclass with a dedicated `test()' method.  Basically, you would
# invoke the DD test case minimization algorithm (= the `ddmin()'
# method) with a list of characters; the `test()' method would combine
# them to a document and run the test.  This should be easy to realize
# and give you some good starting results; the file includes a simple
# sample application.
#
# This file is in the public domain; feel free to copy, modify, use
# and distribute this software as you wish - with one exception.
# Passau University has filed a patent for the use of delta debugging
# on program states (A. Zeller: `Isolating cause-effect chains',
# Saarland University, 2001).  The fact that this file is publicly
# available does not imply that I or anyone else grants you any rights
# related to this patent.
#
# The use of Delta Debugging to isolate failure-inducing code changes
# (A. Zeller: `Yesterday, my program worked', ESEC/FSE 1999) or to
# simplify failure-inducing input (R. Hildebrandt, A. Zeller:
# `Simplifying failure-inducing input', ISSTA 2000) is, as far as I
# know, not covered by any patent, nor will it ever be.  If you use
# this software in any way, I'd appreciate if you include a citation
# such as `This software uses the delta debugging algorithm as
# described in (insert one of the papers above)'.
#
# All about Delta Debugging is found at the delta debugging web site,
#
#               http://www.st.cs.uni-sb.de/dd/
#
# Happy debugging,
#
# Andreas Zeller


import logging
import numpy as np
from typing import List, Tuple
import random
import math
from numpy import ndarray
import networkx as nx
from networkx import DiGraph
from tabulate import tabulate
from conf import DATA_FILE

from collections import deque
from typing import List
import copy

from model.build_result import BuildResult
from model.model import LOG_MATRIX_MODEL, MATRIX_MODEL, LOG_MODEL, NON_MODEL

logger = logging.getLogger()


def write_data(mes):
    try:
        with open(DATA_FILE, 'a') as f:
            f.write(mes)
    except Exception as e:
        logger.warning(str(e))


# Start with some helpers.
class OutcomeCache:
    # This class holds test outcomes for configurations.  This avoids
    # running the same test twice.

    # The outcome cache is implemented as a tree.  Each node points
    # to the outcome of the remaining list.
    #
    # Example: ([1, 2, 3], PASS), ([1, 2], FAIL), ([1, 4, 5], FAIL):
    #
    #      (2, FAIL)--(3, PASS)
    #     /
    # (1, None)
    #     \
    #      (4, None)--(5, FAIL)

    def __init__(self):
        self.tail = {}  # Points to outcome of tail
        self.result = None  # Result so far

    def add(self, c, result):
        """Add (C, RESULT) to the cache.  C must be a list of scalars."""
        cs = c[:]
        cs.sort()

        p = self
        for start in range(len(c)):
            if c[start] not in p.tail:
                p.tail[c[start]] = OutcomeCache()
            p = p.tail[c[start]]

        p.result = result

    def lookup(self, c):
        """Return RESULT if (C, RESULT) is in the cache; None, otherwise."""
        p = self
        for start in range(len(c)):
            if c[start] not in p.tail:
                return None
            p = p.tail[c[start]]

        return p.result

    def lookup_superset(self, c, start=0):
        """Return RESULT if there is some (C', RESULT) in the cache with
        C' being a superset of C or equal to C.  Otherwise, return None."""

        # FIXME: Make this non-recursive!
        if start >= len(c):
            if self.result:
                return self.result
            elif self.tail != {}:
                # Select some superset
                superset = self.tail[list(self.tail.keys())[0]]
                return superset.lookup_superset(c, start + 1)
            else:
                return None

        if c[start] in self.tail:
            return self.tail[c[start]].lookup_superset(c, start + 1)

        # Let K0 be the largest element in TAIL such that K0 <= C[START]
        k0 = None
        for k in self.tail.keys():
            if (k0 == None or k > k0) and k <= c[start]:
                k0 = k

        if k0 != None:
            return self.tail[k0].lookup_superset(c, start)

        return None

    def lookup_subset(self, c):
        """Return RESULT if there is some (C', RESULT) in the cache with
        C' being a subset of C or equal to C.  Otherwise, return None."""
        p = self
        for start in range(len(c)):
            if c[start] in p.tail:
                p = p.tail[c[start]]

        return p.result


# Test the outcome cache
def oc_test():
    oc = OutcomeCache()

    assert oc.lookup([1, 2, 3]) == None
    oc.add([1, 2, 3], 4)
    assert oc.lookup([1, 2, 3]) == 4
    assert oc.lookup([1, 2, 3, 4]) == None

    assert oc.lookup([5, 6, 7]) == None
    oc.add([5, 6, 7], 8)
    assert oc.lookup([5, 6, 7]) == 8

    assert oc.lookup([]) == None
    oc.add([], 0)
    assert oc.lookup([]) == 0

    assert oc.lookup([1, 2]) == None
    oc.add([1, 2], 3)
    assert oc.lookup([1, 2]) == 3
    assert oc.lookup([1, 2, 3]) == 4

    assert oc.lookup_superset([1]) == 3 or oc.lookup_superset([1]) == 4
    assert oc.lookup_superset([1, 2]) == 3 or oc.lookup_superset([1, 2]) == 4
    assert oc.lookup_superset([5]) == 8
    assert oc.lookup_superset([5, 6]) == 8
    assert oc.lookup_superset([6, 7]) == 8
    assert oc.lookup_superset([7]) == 8
    assert oc.lookup_superset([]) != None

    assert oc.lookup_superset([9]) == None
    assert oc.lookup_superset([7, 9]) == None
    assert oc.lookup_superset([-5, 1]) == None
    assert oc.lookup_superset([1, 2, 3, 9]) == None
    assert oc.lookup_superset([4, 5, 6, 7]) == None

    assert oc.lookup_subset([]) == 0
    assert oc.lookup_subset([1, 2, 3]) == 4
    assert oc.lookup_subset([1, 2, 3, 4]) == 4
    assert oc.lookup_subset([1, 3]) == None
    assert oc.lookup_subset([1, 2]) == 3

    assert oc.lookup_subset([-5, 1]) == None
    assert oc.lookup_subset([-5, 1, 2]) == 3
    assert oc.lookup_subset([-5]) == 0


# Main Delta Debugging algorithm.
class DD:
    # Delta debugging base class.  To use this class for a particular
    # setting, create a subclass with an overloaded `test()' method.
    #
    # Main entry points are:
    # - `ddmin()' which computes a minimal failure-inducing configuration, and
    # - `dd()' which computes a minimal failure-inducing difference.
    #
    # See also the usage sample at the end of this file.
    #
    # For further fine-tuning, you can implement an own `resolve()'
    # method (tries to add or remove configuration elements in case of
    # inconsistencies), or implement an own `split()' method, which
    # allows you to split configurations according to your own
    # criteria.
    #
    # The class includes other previous delta debugging alorithms,
    # which are obsolete now; they are only included for comparison
    # purposes.

    # Test outcomes.
    PASS = "FAIL"
    FAIL = "PASS"
    UNRESOLVED = "UNRESOLVED"
    CE = "CE"
    GREEDY = 0.1
    TOP_k = 10
    TOP_MAX = 2000
    FAILURE_STEP = 0
    MODEL = LOG_MATRIX_MODEL

    # Resolving directions.
    ADD = "ADD"  # Add deltas to resolve
    REMOVE = "REMOVE"  # Remove deltas to resolve

    # Debugging output (set to 1 to enable)
    debug_test = 0
    debug_dd = 0
    debug_split = 0
    debug_resolve = 0
    CE_DICT = {}
    REL_UPD = set()
    GROUP_SEED = [(-0.05, 0), (0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1)]

    def __init__(self):
        self.__resolving = 0
        self.__last_reported_length = 0
        self.monotony = 0
        self.outcome_cache = OutcomeCache()
        self.cache_outcomes = 1
        self.minimize = 1
        self.maximize = 1
        self.assume_axioms_hold = 1
        self.ce_his = set()

    # Helpers
    def __listminus(self, c1, c2):
        """Return a list of all elements of C1 that are not in C2."""
        s2 = {}
        for delta in c2:
            s2[delta] = 1

        c = []
        for delta in c1:
            if delta not in s2:
                c.append(delta)

        return c

    def __listintersect(self, c1, c2):
        """Return the common elements of C1 and C2."""
        s2 = {}
        for delta in c2:
            s2[delta] = 1

        c = []
        for delta in c1:
            if delta in s2:
                c.append(delta)

        return c

    def __listunion(self, c1, c2):
        """Return the union of C1 and C2."""
        s1 = {}
        for delta in c1:
            s1[delta] = 1

        c = c1[:]
        for delta in c2:
            if delta not in s1:
                c.append(delta)

        return c

    def __listsubseteq(self, c1, c2):
        """Return 1 if C1 is a subset or equal to C2."""
        s2 = {}
        for delta in c2:
            s2[delta] = 1

        for delta in c1:
            if delta not in s2:
                return 0

        return 1

    def show_status(self, run, cs, n):
        print("dd (run #" + repr(run) + "): trying", end=' ')
        for i in range(n):
            if i > 0:
                print("+", end=' ')
            print(len(cs[i]), end=' ')
        print()

    # Output
    def coerce(self, c):
        """Return the configuration C as a compact string"""
        # Default: use printable representation
        return repr(c)

    def pretty(self, c):
        """Like coerce(), but sort beforehand"""
        sorted_c = c[:]
        sorted_c.sort(key=str)
        return self.coerce(sorted_c)

    # Testing
    def test(self, c, dest_dir=None, uid=None):
        """Test the configuration C.  Return PASS, FAIL, or UNRESOLVED"""
        c.sort()

        # If we had this test before, return its result
        if self.cache_outcomes:
            cached_result = self.outcome_cache.lookup(c)
            if cached_result != None:
                write_data("test({}) = {} (from test set)\n".format(self.coerce(c), repr(cached_result)))
                return cached_result

        # if self.monotony:
        #     # Check whether we had a passing superset of this test before
        #     cached_result = self.outcome_cache.lookup_superset(c)
        #     if cached_result == self.PASS:
        #         return self.PASS

        #     cached_result = self.outcome_cache.lookup_subset(c)
        #     if cached_result == self.FAIL:
        #         return self.FAIL

        if self.debug_test:
            logger.debug("test({})...".format(self.coerce(c)))

        outcome = self._test(c, dest_dir, uid)

        if self.debug_test:
            logger.debug("test({}) = {}".format(self.coerce(c), repr(outcome)))

        write_data("test({}) = {}\n".format(self.coerce(c), repr(outcome)))

        if self.cache_outcomes:
            self.outcome_cache.add(c, outcome)

        return outcome

    def _test(self, dest_dir, uid, keep_variant):
        """Stub to overload in subclasses"""
        return self.UNRESOLVED  # Placeholder

    def _build(self, c, uid=None, ignore_ref=False, keep_variant=False):
        """Stub to overload in subclasses"""
        return False, None, None, None  # Placeholder

    # Splitting
    def split(self, c, n):
        """Split C into [C_1, C_2, ..., C_n]."""
        if self.debug_split:
            logger.debug("split({}, {})...".format(self.coerce(c), repr(n)))

        outcome = self._split(c, n)

        if self.debug_split:
            logger.debug("split({}, {}) = {}".format(self.coerce(c), repr(n), repr(outcome)))

        return outcome

    def _split(self, c, n):
        """Stub to overload in subclasses"""
        subsets = []
        start = 0
        for i in range(n):
            subset = c[start:start + int((len(c) - start) / (n - i))]
            subsets.append(subset)
            start = start + len(subset)
        return subsets

    # Resolving
    def resolve(self, csub, c, direction):
        """If direction == ADD, resolve inconsistency by adding deltas
           to CSUB.  Otherwise, resolve by removing deltas from CSUB."""

        if self.debug_resolve:
            logger.debug("resolve({}, {}, {})...".format(repr(csub), self.coerce(c), repr(direction)))

        outcome = self._resolve(csub, c, direction)

        if self.debug_resolve:
            logger.debug("resolve({}, {}, {}) = {}".format(repr(csub), self.coerce(c), repr(direction), repr(outcome)))

        return outcome

    def _resolve(self, csub, c, direction):
        """Stub to overload in subclasses."""
        # By default, no way to resolve
        return None

    # Test with fixes
    def test_and_resolve_ddmin(self, csub, r, c, direction):
        """Repeat testing CSUB + R while unresolved."""

        initial_csub = csub[:]
        c2 = self.__listunion(r, c)

        csubr = self.__listunion(csub, r)
        t = self.test(csubr)

        # necessary to use more resolving mechanisms which can reverse each
        # other, can (but needn't) be used in subclasses
        self._resolve_type = 0

        if t == self.UNRESOLVED:
            flag = True
            self.__resolving = 1
            csubr = self.resolve(csubr, c, direction)

            if csubr == None:
                # Nothing left to resolve
                flag = False

            elif len(csubr) >= len(c2):
                # Added everything: csub == c2. ("Upper" Baseline)
                # This has already been tested.
                csubr = None
                flag = False

            elif len(csubr) <= len(r):
                # Removed everything: csub == r. (Baseline)
                # This has already been tested.
                csubr = None
                flag = False
            if flag:
                t = self.test(csubr)

        self.__resolving = 0
        if csubr == None:
            return self.UNRESOLVED, initial_csub

        # assert t == self.PASS or t == self.FAIL
        csub = self.__listminus(csubr, r)
        return t, csub

    # Test with fixes
    def test_and_resolve(self, csub, r, c, direction):
        """Repeat testing CSUB + R while unresolved."""

        initial_csub = csub[:]
        c2 = self.__listunion(r, c)

        csubr = self.__listunion(csub, r)
        t = self.test(csubr)

        # necessary to use more resolving mechanisms which can reverse each
        # other, can (but needn't) be used in subclasses
        self._resolve_type = 0

        while t == self.UNRESOLVED:
            self.__resolving = 1
            csubr = self.resolve(csubr, c, direction)

            if csubr == None:
                # Nothing left to resolve
                break

            if len(csubr) >= len(c2):
                # Added everything: csub == c2. ("Upper" Baseline)
                # This has already been tested.
                csubr = None
                break

            if len(csubr) <= len(r):
                # Removed everything: csub == r. (Baseline)
                # This has already been tested.
                csubr = None
                break

            t = self.test(csubr)

        self.__resolving = 0
        if csubr == None:
            return self.UNRESOLVED, initial_csub

        # assert t == self.PASS or t == self.FAIL
        csub = self.__listminus(csubr, r)
        return t, csub

    # Inquiries
    def resolving(self):
        """Return 1 while resolving."""
        return self.__resolving

    # Logging
    def report_progress(self, c, title):
        if len(c) != self.__last_reported_length:
            logger.info('{}: {} deltas left: {}'.format(title, repr(len(c)), self.coerce(c)))
            # print(title + ": " + repr(len(c)) + " deltas left:", self.coerce(c))
            self.__last_reported_length = len(c)

    # def test_mix(self, csub, c, direction):
    #     t = self.test(csub)
    #     if t != self.UNRESOLVED:
    #         return (t, csub)

    #     csubr = self.resolve(csub, c, direction)
    #     if csubr == None or len(csubr) == 0:
    #         return (t, csub)
    #     t = self.test(csubr)
    #     if t != self.UNRESOLVED:
    #         return (t, csubr)
    #     else:
    #         return (t, csub)

    def test_mix(self, csub, c, direction):
        if self.minimize:
            (t, csub) = self.test_and_resolve_ddmin(csub, [], c, direction)
            if t == self.FAIL:
                return (t, csub)

        if self.maximize:
            csubbar = self.__listminus(self.CC, csub)
            cbar = self.__listminus(self.CC, c)
            if direction == self.ADD:
                directionbar = self.REMOVE
            else:
                directionbar = self.ADD

            (tbar, csubbar) = self.test_and_resolve_ddmin(csubbar, [], cbar,
                                                          directionbar)

            csub = self.__listminus(self.CC, csubbar)

            if tbar == self.PASS:
                t = self.FAIL
            elif tbar == self.FAIL:
                t = self.PASS
            else:
                t = self.UNRESOLVED

        return (t, csub)

    # Delta Debugging (new ISSTA version)
    def ddgen(self, c, minimize, maximize):
        """Return a 1-minimal failing subset of C"""
        self.minimize = minimize
        self.maximize = maximize

        n = 2
        self.CC = c

        if self.debug_dd:
            logger.debug("dd({}, {})...".format(self.pretty(c), repr(n)))
        #  matrix = self._get_dep_matrix()
        #
        #  self.set_tokens2hunks(cids=c)
        #
        #  print(tabulate(matrix, tablefmt="fancy_grid", showindex=True, headers=list(range(len(c)))))
        #  outcome = self.reldd(c,matrix)Ï
        outcome = self._dd(c, n)
        print(f"cc{len(outcome)}")

        if self.debug_dd:
            logger.debug("dd({}, {}) = {}".format(self.pretty(c), repr(n), repr(outcome)))

        return outcome

    def _get_dep_matrix(self):
        return None

    def getIdx2test(self, inp1, inp2):
        res = []
        for elm in inp1:
            if not (elm in inp2):
                res.append(elm)
        return res

    def computRatio(self, deleteconfig, p) -> float:
        res = 0
        tmplog = 0.0
        for delc in deleteconfig:
            if 0 < p[delc] < 1:
                tmplog += math.log(1 - p[delc])
        tmplog = math.exp(tmplog)
        res = 1 / (1 - tmplog)
        return res

    def sample(self, p):
        delset = []
        idx = np.argsort(p)  # sort by probabilities and return index
        k = 0
        tmp = 1
        last = -99999
        idxlist = list(idx)
        i = 0
        while i < len(p):
            if p[idxlist[i]] == 0:
                k = k + 1
                i = i + 1
                continue
            if not p[idxlist[i]] < 1:
                break
            for j in range(k, i + 1):
                tmp *= (1 - p[idxlist[j]])
            tmp *= (i - k + 1)
            if tmp < last:
                break
            last = tmp
            tmp = 1
            i = i + 1
        while i > k:
            i = i - 1
            delset.append(idxlist[i])
        return delset

    def testDone(self, p):
        return all(x >= 1 or x <= 0 for x in p)

    def sample_x(self, p: list, retIdx: list, test_count_map: dict, use_cc: bool):
        delIdx = self.sample(p)

        idx2test = self.getIdx2test(retIdx, delIdx)
        if len(idx2test) == 0 or use_cc:
            sliced_map = {key: test_count_map[key] for key in retIdx}
            sorted_keys = [k for k, v in
                           sorted(sliced_map.items(), key=lambda x: x[1], reverse=True)]  # 使用 reverse=True 进行倒序排序
            idx = int(len(retIdx) * 0.25)
            if idx <= 0:
                idx = 1
                # 获取前25%的索引位置
            idx2test = sorted_keys[:idx]

        if self.outcome_cache.lookup(sorted(idx2test)) != None:
            p_s = [p[i] for i in retIdx]
            idx2test = self.random_selection(retIdx, p_s)
        if len(idx2test) == 0 or len(idx2test) == len(retIdx):
            p_s = [p[i] for i in retIdx]
            idx2test = self.random_selection(retIdx, p_s)
        idx2test = list(set(idx2test))

        delIdx = self.getIdx2test(retIdx, idx2test)
        return sorted(idx2test), sorted(delIdx)

    def put_test_count(self, idx2test, test_count_map):
        for num in idx2test:
            if num in test_count_map:
                test_count_map[num] += 1
            else:
                test_count_map[num] = 1

    def test_mix_prodd(self, csub, c):
        if self.MODEL == LOG_MODEL or self.MODEL == NON_MODEL:
            return self.UNRESOLVED, csub
        (t, csub) = self.test_and_resolve(csub, [], c, self.ADD)
        if t != self.UNRESOLVED:
            return (t, csub)
        else:
            (t, csub) = self.test_and_resolve(csub, [], c, self.REMOVE)
        return (t, csub)

    def _prodd(self, c):
        print("Use ProbDD")
        assert self.test([]) == self.PASS  # check whether F meet T
        run = 1
        p = []
        for idx in range(0, len(c)):  # initialize the probability for each element in the input sequence
            p.append(0.1)
        while not self.testDone(p):

            delIdx = self.sample(p)
            if len(delIdx) == 0:
                break
            idx2test = self.getIdx2test(c, delIdx)
            res = self.test(idx2test)

            if res == self.FAIL:  # set probabilities of the deleted elements to 0
                for set0 in range(0, len(p)):
                    if set0 not in idx2test:
                        p[set0] = 0
                c = idx2test
            else:  # test(seq2test, *test_args) == PASS:
                p_cp = p[:]
                for setd in range(0, len(p)):
                    if setd in delIdx and 0 < p[setd] < 1:
                        delta = (self.computRatio(delIdx, p_cp) - 1) * p_cp[setd]
                        p[setd] = p_cp[setd] + delta
            run = run + 1
            write_data("p: " + repr(p) + "\n")
            print(f"{idx2test}:{res}")
            print("p: " + repr(p))
        write_data('loop time: {}\n'.format(run))
        return c

    def _reldd(self, c, matrix: ndarray):
        print("Use RelDD")
        # assert self.test([]) == self.PASS #check wether F meet T
        retIdx = c[:]
        p = []
        his = set()
        test_count_map = {}
        for idx in range(0, len(c)):  # initialize the probability for each element in the input sequence
            p.append(0.1)
            test_count_map[idx] = 0
        last_p = []
        cc = 0
        falure_step = 0
        step = 0
        iscompile = True
        random.seed(42)
        while (not self.testDone(p)):
            if len(retIdx) == 1:
                break
            if step == 0:  # 给定一个起始点
                idx2test = self.select_start_loc(c)
                if len(idx2test) == len(retIdx):
                    idx2test, delIdx = self.sample_x(p, retIdx, test_count_map, False)
                delIdx = self.getIdx2test(retIdx, idx2test)
            else:
                use_cc = self.make_decision_by_pro(falure_step / len(retIdx))
                if use_cc:
                    falure_step = 0
                idx2test, delIdx = self.sample_x(p, retIdx, test_count_map, use_cc)

            step += 1
            # delIdx = self.sample(p)
            if len(delIdx) == 0:
                print("err:delIDX ==0")
                break
            res = self.predict_result(idx2test)
            fix_set = None
            is_fix = False
            if res != self.PASS:

                iscompile, dest_dir, uid, build_result = self.check_compile(idx2test, None, None, retIdx, matrix)
                if not iscompile:
                    fix_set = idx2test[:]
                    fix_set, iscompile, dest_dir, uid = self.try_fix_with_gen(idx2test=idx2test, retIdx=retIdx,
                                                                              matrix=matrix,
                                                                              last_build_result=build_result, p=p)
                    if len(fix_set) == len(retIdx):
                        iscompile = False
                    if not iscompile:
                        fix_set, iscompile, dest_dir, uid = self.select_consider_point(idx2test, retIdx, matrix,
                                                                                       test_count_map)
                        if len(fix_set) == len(retIdx):
                            iscompile = False
                        self.TOP_k = self.TOP_k + 10
                        if self.TOP_k > self.TOP_MAX:
                            self.TOP_k = self.TOP_MAX
                    if iscompile:
                        is_fix = True
                    fix_set.sort()
                idx2test.sort()
                idx2test_back = idx2test[:]
                if is_fix:
                    idx2test = fix_set
                delIdx = self.getIdx2test(retIdx, idx2test)
                res = self.test(idx2test, dest_dir, uid)
                if res != self.FAIL:
                    if 0 < len(delIdx) < 3:
                        res_b = self.test(delIdx)
                        temp = idx2test[:]
                        if len(idx2test) != 0 and res_b == self.FAIL:
                            idx2test = sorted(delIdx)
                            delIdx = temp
                            res = res_b
            else:
                self.outcome_cache.add(sorted(idx2test), self.PASS)
            
            self.put_test_count(idx2test=idx2test, test_count_map=test_count_map)
            print('{}:{}'.format(idx2test, res))
            his.add(self.get_list_str(idx2test))
            if res == self.FAIL:  # set probabilities of the deleted elements to 0
                for set0 in range(0, len(p)):
                    if set0 not in idx2test:
                        p[set0] = 0
                        for index in range(0, len(matrix)):
                            matrix[index][set0] = 0.0
                            matrix[set0][index] = 0.0
                retIdx = idx2test
                falure_step = 0
            else:  # test(seq2test, *test_args) == PASS:
                is_fix_by_add = set(idx2test_back).issubset(set(idx2test))
                if is_fix and is_fix_by_add:
                    idx2test = idx2test_back
                    delIdx = self.getIdx2test(retIdx, idx2test)
                last_p = p[:]
                gtflag = all(x > 0.5 for x in last_p if x != 0)
                for setd in range(0, len(p)):
                    if setd in delIdx and 0 < p[setd] < 1:
                        delta = (self.computRatio(delIdx, last_p) - 1) * last_p[setd]
                        p[setd] = last_p[setd] + delta 
                        if p[setd] > 0.95 and not gtflag:
                           left = last_p[setd]
                           if left >= 0.95:
                               left =0.9
                           p[setd] = random.uniform(left, 0.95)
                falure_step += 1

            if set(last_p) == set(p):
                cc += 1
            if cc > 100:
                break
            print(p)
            # print(tabulate(matrix, tablefmt="fancy_grid", showindex=True, headers=list(range(len(p)))))
        print("{}->{}->{}".format(len(his), len(self.CE_DICT), retIdx))
        write_data("{}->{}->{}".format(len(his), len(self.CE_DICT), retIdx))
        return retIdx

    def try_fix_with_gen(self, idx2test: list, retIdx: list, matrix: ndarray, last_build_result, p):
        # 分两种修复方式，log-based和matrix-based
        # 首先计算切片矩阵信息熵
        rate = 1
        if self.MODEL != LOG_MATRIX_MODEL:
            rate = 2
        idx2test.sort()
        hx = self.calculate_slice_entropy(matrix=matrix, row_indices=idx2test, col_indices=retIdx)
        entropy_list = []

        gens_by_matrix = self.gen_fix_by_matrix(idx2test, retIdx, matrix, rate * self.TOP_k)

        for item_list in gens_by_matrix:
            if item_list == None or len(item_list) == 0:
                continue
            item_list_ = self.getIdx2test(retIdx, item_list)
            entropy_list_score = self.calculate_slice_entropy(matrix, item_list, item_list_)
            entropy_list.append(entropy_list_score)

        sorted_index_list = [x for _, x in sorted(zip(entropy_list, gens_by_matrix))][:rate * self.TOP_k]
        sorted_index_list = [item for item in sorted_index_list if self.outcome_cache.lookup(sorted(item)) == None]

        if hx < 1.5:
            print("try fix with matrix")
            (t, csub) = self.test_mix_prodd(idx2test, retIdx)
            csub = list(set(csub))
            if t != self.UNRESOLVED:
                return csub, True, None, None
            else:
                # 处理lastbuildinfo
                for item in sorted_index_list:
                    item = list(set(item))
                    iscompile, dest_dir, uid, build_result = self.check_compile(item, None, None, retIdx, matrix)
                    if iscompile:
                        return item, iscompile, dest_dir, uid

                print("try fix with log")
                gens_by_logs = self.gen_fix_by_log(idx2test, retIdx, last_build_result, rate * self.TOP_k, p)
                gens_by_logs = [item for item in gens_by_logs if self.outcome_cache.lookup(sorted(item)) == None]
                for item in gens_by_logs:
                    item = list(set(item))
                    iscompile, dest_dir, uid, build_result = self.check_compile(item, None, None, retIdx, matrix)
                    if iscompile:
                        return item, iscompile, dest_dir, uid
        else:
            print("try fix with log")
            gens_by_logs = self.gen_fix_by_log(idx2test, retIdx, last_build_result, rate * self.TOP_k, p)
            gens_by_logs = [item for item in gens_by_logs if self.outcome_cache.lookup(sorted(item)) == None]
            for item in gens_by_logs:
                item = list(set(item))
                iscompile, dest_dir, uid, build_result = self.check_compile(item, None, None, retIdx, matrix)
                if iscompile:
                    return item, iscompile, dest_dir, uid

            print("try fix with matrix")
            (t, csub) = self.test_mix_prodd(idx2test, retIdx)
            if t != self.UNRESOLVED:
                return csub, True, None, None
            else:
                for item in sorted_index_list:
                    item = list(set(item))
                    iscompile, dest_dir, uid, build_result = self.check_compile(item, None, None, retIdx, matrix)
                    if iscompile:
                        return item, iscompile, dest_dir, uid

        return idx2test, False, None, None

    def select_consider_point(self, idx2test: list, retIdx: list, matrix: ndarray, test_count_map: dict):
        retIdx = sorted(retIdx)
        max_values = np.max(matrix, axis=0)
        groups = [[] for _ in range(len(self.GROUP_SEED))]

        for i, value in enumerate(max_values):
            for j, (start, end) in enumerate(self.GROUP_SEED):
                if start < value <= end:
                    groups[j].append(i)
                    break
        groups = [list(set(group) & set(retIdx)) for group in groups]

        groups = [item for item in groups if len(item) > 0]

        subsets = self.generate_subsets_group(groups)
        subsets = [subset for subset in subsets if sorted(subset) != retIdx and len(subset) != 0]
        subsets.append(self.getIdx2test(retIdx, idx2test))

        min_value = min(test_count_map.values())
        min_elements = [key for key, value in test_count_map.items() if value == min_value]
        if len(min_elements) != len(retIdx):
            subsets.append(min_elements)
        for sub in subsets:
            if len(sub) == 0 or len(sub) == len(retIdx) or self.get_list_str(sub) not in self.CE_DICT:
                continue
            iscompile, dest_dir, uid, build_result = self.check_compile(sub, None, None, retIdx, matrix)
            if iscompile:
                return sub, iscompile, dest_dir, uid
        return idx2test, False, None, None

    def generate_subsets_group(self, lst):
        n = len(lst)
        subsets = []
        for i in range(1, 2 ** n):  # 从1到2^n-1遍历所有可能的子集表示整数，跳过空集表示的整数0
            subset = []
            for j in range(n):
                if (i >> j) & 1:  # 检查当前位是否为1
                    subset.extend(lst[j])  # 将对应位置的元素添加到子集中
            subsets.append(subset)

        return subsets

    def predict_result(self, idx2test):
        idx2test.sort()
        res = self.outcome_cache.lookup_superset(idx2test)
        if res == self.PASS:
            return self.PASS
        else:
            return self.UNRESOLVED

    def gen_fix_by_log(self, idx2test: list, retIdx: list, last_build_result: BuildResult, max: int, p: list) -> List[
        List]:
        # 根据 log中的信息进行上述或者添加
        idx2test_back = idx2test[:]
        result = []
        err_cids = last_build_result.rcids[:]
        if self.MODEL == MATRIX_MODEL:
            return result
        err_cids = list(set(err_cids) & set(retIdx))
        if len(err_cids) == 0:
            err_cids = list(set(retIdx) - set(idx2test_back))
            for i in range(0, max):
                p_s = [p[l] for l in idx2test_back]
                res = self.random_selection(idx2test_back[:], p_s)
                p_d = [p[m] for m in err_cids]
                res.extend(self.random_selection(err_cids[:], p_d))
            result.append(res)
            return result

        result.append(list(set(idx2test_back) | set(err_cids)))
        result.append(list(set(idx2test_back) - set(err_cids)))

        for item in range(2, len(err_cids) + 1):
            n = len(err_cids) // item
            results = [err_cids[i:i + n] for i in range(0, len(err_cids), n)]
            for list_item in results:
                add_fix_list = list(set(idx2test_back) | set(list_item))
                add_fix_list.sort()
                del_fix_list = list(set(idx2test_back) - set(list_item))
                del_fix_list.sort()
                result.append(add_fix_list)
                result.append(del_fix_list)
                if len(result) >= max:
                    return result
        result.append(list(set(err_cids)))

        return result

    def gen_fix_by_matrix(self, idx2test: list, retIdx: list, matrtix: ndarray, max) -> List[List]:
        # 根据 martix中的信息进行上述或者添加
        # 首先尝试添加
        idx2test.sort()
        result = []
        if self.MODEL == LOG_MODEL:
            return result
        delIdx = self.getIdx2test(retIdx, idx2test)
        slice_matrix = matrtix[idx2test][:, delIdx]
        pro4del = np.max(slice_matrix, axis=0)
        pro4Idx = np.max(slice_matrix, axis=1)
        for i in range(0, max):
            gen_set = set()
            sub = idx2test[:]
            sub_ = delIdx[:]
            while len(sub_) > 0:  # add
                sub = self.select_next_nodes(sub, sub_, matrtix)
                if len(sub) > 0:
                    sub_ = self.getIdx2test(sub_, sub)
                    gen_set.update(sub)
                else:
                    break
            if len(gen_set) != len(delIdx) and len(gen_set) > 0:
                gen_list = sorted(list(set(gen_set) | set(idx2test)))
                if not self.is_list_in_nested_list(result, gen_list) and self.get_list_str(
                        gen_list) not in self.CE_DICT:
                    result.append(gen_list)

                    # rand_add_set = self.random_selection(delIdx, pro4del)
            # if len(rand_add_set)!= len(delIdx):
            #     gen2_list = list(set(idx2test)|set(rand_add_set))
            #     if not self.is_list_in_nested_list(result,gen2_list) and self.get_list_str(gen2_list) not in self.CE_DICT:
            #             result.append(gen2_list)

            sub = idx2test[:]
            sub_ = delIdx[:]
            del_nodes = self.select_del_nodes(sub, sub_, matrtix)

            if 0 < len(del_nodes) < len(idx2test):
                del_list = sorted(list(set(idx2test) - del_nodes))
                if not self.is_list_in_nested_list(result, del_list) and self.get_list_str(
                        del_list) not in self.CE_DICT:
                    result.append(del_list)

                    # rand_del_set = self.random_selection(idx2test, pro4Idx)
            # if 0< len(rand_del_set)<len(idx2test):
            #     gen3_list = sorted(list(set(idx2test)-set(rand_del_set)))
            #     if not self.is_list_in_nested_list(result,gen3_list) and self.get_list_str(gen3_list) not in self.CE_DICT:
            #         result.append(gen3_list)

        return result

    def random_selection(self, set_data, probabilities):
        right = len(set_data) - 1
        count_nonzero = sum(1 for x in probabilities if x != 0)-1
        if right > count_nonzero:
            right = count_nonzero
        if right <= 0:
            return set_data
        if right == 1:
            size = 1
        else:
            size = np.random.randint(1, right)  # 随机选择子集的大小
        total_sum = sum(probabilities)
        probabilities = [prob / total_sum for prob in probabilities]
        selected_indices = np.random.choice(len(set_data), size, replace=False, p=probabilities)  # 根据概率随机选择索引
        selected_data = [set_data[i] for i in selected_indices]  # 根据索引获取选择的数据
        return selected_data

    def is_list_in_nested_list(self, nested_list, target_list):
        for sublist in nested_list:
            if sublist == target_list:
                return True
        return False

    def generate_subsets(self, my_set, threshold):
        subsets = []

        def generate_helper(subset, index):
            nonlocal subsets
            if len(subsets) >= threshold:
                return
            if len(subset) > 0 and len(subset) < len(my_set):
                subsets.append(subset)

            for i in range(index, len(my_set)):
                generate_helper(subset + [my_set[i]], i + 1)

        generate_helper([], 0)

        return subsets

    def select_del_nodes(self, sub: list, sub_: list, matrtix: ndarray):
        delset = set()
        sliced_matrix = matrtix[sub][:, sub_]
        max_values = np.max(sliced_matrix, axis=1)
        if np.all(max_values < 0.5):
            max_values = max_values + 0.25
        for i in range(0, len(max_values)):
            if self.make_decision_by_pro(max_values[i]):
                delset.add(sub[i])
        return delset

    def make_decision_by_pro(self, pro) -> bool:
        decision = np.random.choice([False, True], p=[1 - pro, pro])
        return decision

    def select_next_nodes(self, sub: list, sub_: list, matrtix: ndarray):
        addset = set()
        sliced_matrix = matrtix[sub][:, sub_]
        max_values = np.max(sliced_matrix, axis=0)
        if np.all(max_values < 0.5):
            max_values = max_values + 0.25
        for i in range(0, len(max_values)):
            if self.make_decision_by_pro(max_values[i]):
                addset.add(sub_[i])
        return list(addset)

    def calculate_slice_entropy(self, matrix, row_indices: list, col_indices: list):
        slice_matrix = matrix[row_indices][:, col_indices]

        entropy = -np.sum(slice_matrix * np.log2(slice_matrix + 1e-10))  # 避免log2(0)的错误，加上一个小值

        return entropy

    def updateMatrix(self, testIdx: list, delIdx: list, matrix: ndarray):
        ss = f"{self.get_list_str(testIdx)}->{self.get_list_str(delIdx)}"
        if ss in self.REL_UPD:
            return
        self.REL_UPD.add(ss)
        tmplog = 0.00
        for itemt in testIdx:
            for itemd in delIdx:
                if 0 < matrix[itemt][itemd] < 1:
                    tmplog += math.log(1.0 - matrix[itemt][itemd])
        if tmplog == 0:
            return
        tmplog = math.pow(math.e, tmplog)
        print("update {}->{}".format(testIdx, delIdx))
        for itemt in testIdx:
            for itemd in delIdx:
                if matrix[itemt][itemd] != 0:
                    matrix[itemt][itemd] = min(matrix[itemt][itemd] / (1.0 - tmplog), 1.0)
                    if matrix[itemt][itemd] >= 0.99:
                        matrix[itemt][itemd] = 1.0

    def is_matrix_binary(self, m):
        arr = np.array(m)
        return np.all((arr == 0) | (arr == 1))

    def getCESubsets(self, testset: list, cemap: dict):
        res = []
        teststr = self.get_list_str(testset)
        for key, value in cemap.items():
            if key in teststr:
                res.append(value[0])
        return res

    def reset(self, p, matrix):
        for i in range(len(p)):
            if p[i] != 0:
                p[i] = 0.1
        self.resetMatrix(matrix)

    def resetMatrix(self, matrix):
        return None

    def get_list_str(self, t: list):
        if len(t) == 0:
            return ''
        t = list(t)
        t.sort()
        return ''.join(str(i) + '+' for i in t)

    def check_compile(self, n_testIdx, last_testIdx, last_buildInfo, retIdx, matrix):
        err_key = self.get_list_str(n_testIdx)
        if err_key in self.CE_DICT and self.CE_DICT[err_key] != None:
            self.updateMatrix(n_testIdx, self.getIdx2test(retIdx, n_testIdx), matrix)
            return False, None, None, self.CE_DICT[self.get_list_str(n_testIdx)][1]
        t = self.outcome_cache.lookup(n_testIdx)
        if t != None and t != self.UNRESOLVED:
            return True, None, None, None
        is_compile, dtest_dir, uid, build_info = self._build(n_testIdx)
        if not is_compile:
            self.CE_DICT[err_key] = (n_testIdx, build_info)
        # 根据编译结果根新概率矩阵
        if (not is_compile):
            if build_info and last_buildInfo and len(build_info.err_set) < len(last_buildInfo.err_set) and last_testIdx:
                # 更新概率
                if (n_testIdx < last_testIdx):
                    self.updateMatrix(n_testIdx, self.getIdx2test(last_testIdx, n_testIdx), matrix)
            else:
                self.updateMatrix(n_testIdx, self.getIdx2test(retIdx, n_testIdx), matrix)

        if is_compile:  # testIdx 和 delIdx之间的关系被解除
            for item in n_testIdx:
                for itemj in self.getIdx2test(retIdx, n_testIdx):
                    matrix[item][itemj] = 0

        return is_compile, dtest_dir, uid, build_info

    def _dd(self, c, n):
        """Stub to overload in subclasses"""

        assert self.test([]) == self.PASS

        run = 1
        cbar_offset = 0

        # We replace the tail recursion from the paper by a loop
        while 1:
            tc = self.test(c)
            assert tc == self.FAIL or tc == self.UNRESOLVED

            if n > len(c):
                # No further minimizing
                logger.info("dd: done")
                return c

            self.report_progress(c, "dd")

            cs = self.split(c, n)

            self.show_status(run, cs, n)

            c_failed = 0
            cbar_failed = 0

            next_c = c[:]
            next_n = n

            # Check subsets
            for i in range(n):
                if self.debug_dd:
                    logger.debug("dd: trying {}".format(self.pretty(cs[i])))

                (t, cs[i]) = self.test_mix(cs[i], c, self.REMOVE)

                if t == self.FAIL:
                    # Found
                    if self.debug_dd:
                        logger.debug("dd: found {} deltas:".format(len(cs[i])))
                        logger.debug(self.pretty(cs[i]))

                    c_failed = 1
                    next_c = cs[i]
                    next_n = 2
                    cbar_offset = 0
                    self.report_progress(next_c, "dd")
                    break

            if not c_failed:
                # Check complements
                cbars = n * [self.UNRESOLVED]

                # print("cbar_offset =", cbar_offset)

                for j in range(n):
                    i = (j + cbar_offset) % n
                    cbars[i] = self.__listminus(c, cs[i])
                    t, cbars[i] = self.test_mix(cbars[i], c, self.ADD)

                    doubled = self.__listintersect(cbars[i], cs[i])
                    if doubled != []:
                        cs[i] = self.__listminus(cs[i], doubled)

                    if t == self.FAIL:
                        if self.debug_dd:
                            logger.debug("dd: reduced to {}".format(len(cbars[i])))
                            logger.debug("deltas:")
                            logger.debug(self.pretty(cbars[i]))

                        cbar_failed = 1
                        next_c = self.__listintersect(next_c, cbars[i])
                        next_n = next_n - 1
                        self.report_progress(next_c, "dd")

                        # In next run, start removing the following subset
                        cbar_offset = i
                        break

            if not c_failed and not cbar_failed:
                if n >= len(c):
                    # No further minimizing
                    logger.info("dd: done")
                    return c

                next_n = min(len(c), n * 2)
                logger.info("dd: increase granularity to {}".format(next_n))
                cbar_offset = int((cbar_offset * next_n) / n)

            c = next_c
            n = next_n
            run = run + 1
        write_data('loop time: {}\n'.format(run))

    def ddmin(self, c):
        return self.ddgen(c, 1, 0)

    def ddmax(self, c):
        return self.ddgen(c, 0, 1)

    def ddmix(self, c):
        return self.ddgen(c, 1, 1)

    def prodd(self, c):

        n = 2
        self.CC = c

        if self.debug_dd:
            logger.debug("dd({}, {})...".format(self.pretty(c), repr(n)))

        outcome = self._prodd(c)

        if self.debug_dd:
            logger.debug("dd({}, {}) = {}".format(self.pretty(c), repr(n), repr(outcome)))

        return outcome

    def reldd(self, c):
        n = 2
        self.CC = c

        if self.debug_dd:
            logger.debug("dd({}, {})...".format(self.pretty(c), repr(n)))
        matrix = self._get_dep_matrix()

        self.set_tokens2hunks(cids=c)
        # print(tabulate(matrix, tablefmt="fancy_grid", showindex=True, headers=list(range(len(c)))))
        outcome = self._reldd(c, matrix)
        print(f"cc{len(outcome)}")

        if self.debug_dd:
            logger.debug("dd({}, {}) = {}".format(self.pretty(c), repr(n), repr(outcome)))

        return outcome

    # General delta debugging (new TSE version)
    def dddiff(self, c):
        n = 2

        if self.debug_dd:
            logger.debug("dddiff({}, {})...".format(self.pretty(c), repr(n)))

        outcome = self._dddiff([], c, n)

        if self.debug_dd:
            logger.debug("dddiff({}, {}) = {}".format(self.pretty(c), repr(n), repr(outcome)))

        return outcome

    def _dddiff(self, c1, c2, n):
        run = 1
        cbar_offset = 0

        # We replace the tail recursion from the paper by a loop
        while 1:
            if self.debug_dd:
                logger.debug("dd: c1 = {}".format(self.pretty(c1)))
                logger.debug("dd: c2 = {}".format(self.pretty(c2)))

            if self.assume_axioms_hold:
                t1 = self.PASS
                t2 = self.FAIL
            else:
                t1 = self.test(c1)
                t2 = self.test(c2)

            assert t1 == self.PASS
            assert t2 == self.FAIL
            assert self.__listsubseteq(c1, c2)

            c = self.__listminus(c2, c1)

            if self.debug_dd:
                logger.debug("dd: c2 - c1 = {}".format(self.pretty(c)))

            if n > len(c):
                # No further minimizing
                logger.info("dd: done")
                return (c, c1, c2)

            self.report_progress(c, "dd")

            cs = self.split(c, n)

            self.show_status(run, cs, n)

            progress = 0

            next_c1 = c1[:]
            next_c2 = c2[:]
            next_n = n

            # Check subsets
            for j in range(n):
                i = (j + cbar_offset) % n

                if self.debug_dd:
                    logger.debug("dd: trying {}".format(self.pretty(cs[i])))

                (t, csub) = self.test_and_resolve(cs[i], c1, c, self.REMOVE)
                csub = self.__listunion(c1, csub)

                if t == self.FAIL and t1 == self.PASS:
                    # Found
                    progress = 1
                    next_c2 = csub
                    next_n = 2
                    cbar_offset = 0

                    if self.debug_dd:
                        logger.debug("dd: reduce c2 to {} deltas:".format(len(next_c2)))
                        logger.debug(self.pretty(next_c2))
                    break

                if t == self.PASS and t2 == self.FAIL:
                    # Reduce to complement
                    progress = 1
                    next_c1 = csub
                    next_n = max(next_n - 1, 2)
                    cbar_offset = i

                    if self.debug_dd:
                        logger.debug("dd: increase c1 to {} deltas:".format(len(next_c1)))
                        logger.debug(self.pretty(next_c1))
                    break

                csub = self.__listminus(c, cs[i])
                (t, csub) = self.test_and_resolve(csub, c1, c, self.ADD)
                csub = self.__listunion(c1, csub)

                if t == self.PASS and t2 == self.FAIL:
                    # Found
                    progress = 1
                    next_c1 = csub
                    next_n = 2
                    cbar_offset = 0

                    if self.debug_dd:
                        logger.debug("dd: increase c1 to {} deltas:".format(len(next_c1)))
                        logger.debug(self.pretty(next_c1))
                    break

                if t == self.FAIL and t1 == self.PASS:
                    # Increase
                    progress = 1
                    next_c2 = csub
                    next_n = max(next_n - 1, 2)
                    cbar_offset = i

                    if self.debug_dd:
                        logger.debug("dd: reduce c2 to {} deltas:".format(len(next_c2)))
                        logger.debug(self.pretty(next_c2))
                    break

            if progress:
                self.report_progress(self.__listminus(next_c2, next_c1), "dd")
            else:
                if n >= len(c):
                    # No further minimizing
                    logger.info("dd: done")
                    return (c, c1, c2)

                next_n = min(len(c), n * 2)
                logger.info("dd: increase granularity to {}".format(next_n))
                cbar_offset = int((cbar_offset * next_n) / n)

            c1 = next_c1
            c2 = next_c2
            n = next_n
            run = run + 1

    def dd(self, c):
        return self.dddiff(c)  # Backwards compatibility


if __name__ == '__main__':
    # Test the outcome cache
    oc_test()


    # Define our own DD class, with its own test method
    class MyDD(DD):
        def _test_a(self, c):
            "Test the configuration C.  Return PASS, FAIL, or UNRESOLVED."

            # Just a sample
            # if 2 in c and not 3 in c:
            #   return self.UNRESOLVED
            # if 3 in c and not 7 in c:
            #   return self.UNRESOLVED
            if 7 in c and not 2 in c:
                return self.UNRESOLVED
            if 5 in c and 8 in c:
                return self.FAIL
            return self.PASS

        def _test_b(self, c):
            if c == []:
                return self.PASS
            if 1 in c and 2 in c and 3 in c and 4 in c and \
                    5 in c and 6 in c and 7 in c and 8 in c:
                return self.FAIL
            return self.UNRESOLVED

        def _test_c(self, c):
            if 1 in c and 2 in c and 3 in c and 4 in c and \
                    6 in c and 8 in c:
                if 5 in c and 7 in c:
                    return self.UNRESOLVED
                else:
                    return self.FAIL
            if 1 in c or 2 in c or 3 in c or 4 in c or \
                    6 in c or 8 in c:
                return self.UNRESOLVED
            return self.PASS

        def __init__(self):
            self._test = self._test_c
            DD.__init__(self)


    logger.info("WYNOT - a tool for delta debugging.")
    mydd = MyDD()
    # mydd.debug_test     = 1           # Enable debugging output
    # mydd.debug_dd       = 1           # Enable debugging output
    # mydd.debug_split    = 1           # Enable debugging output
    # mydd.debug_resolve  = 1           # Enable debugging output

    # mydd.cache_outcomes = 0
    # mydd.monotony = 0

    logger.info("Minimizing failure-inducing input...")
    c = mydd.ddmin([1, 2, 3, 4, 5, 6, 7, 8])  # Invoke DDMIN
    logger.info("The 1-minimal failure-inducing input is {}".format(c))
    logger.info("Removing any element will make the failure go away.")
    logger.info('')

    logger.info("Computing the failure-inducing difference...")
    (c, c1, c2) = mydd.dd([1, 2, 3, 4, 5, 6, 7, 8])  # Invoke DD
    logger.info("The 1-minimal failure-inducing difference is {}".format(c))
    logger.info("{} passes, {} fails".format(c1, c2))

# Local Variables:
# mode: python
# End:
