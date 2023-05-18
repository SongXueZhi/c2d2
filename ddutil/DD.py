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
from networkx import DiGraph
from tabulate import tabulate

from collections import deque
from typing import List
import copy



logger = logging.getLogger()


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
        self.tail = {}                  # Points to outcome of tail
        self.result = None              # Result so far


    def add(self, c, result):
        """Add (C, RESULT) to the cache.  C must be a list of scalars."""
        cs = c[:]
        cs.sort(key=str)


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


    def lookup_superset(self, c, start = 0):
        """Return RESULT if there is some (C', RESULT) in the cache with
        C' being a superset of C or equal to C.  Otherwise, return None."""


        # FIXME: Make this non-recursive!
        if start >= len(c):
            if self.result:
                return self.result
            elif self.tail != {}:
                # Select some superset
                superset = self.tail[self.tail.keys()[0]]
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
    PASS       = "FAL"
    FAIL       = "PASS"
    UNRESOLVED = "UNRESOLVED"
    CE = "CE"
    GREEDY = 0.1


    # Resolving directions.
    ADD    = "ADD"          # Add deltas to resolve
    REMOVE = "REMOVE"           # Remove deltas to resolve


    # Debugging output (set to 1 to enable)
    debug_test      = 0
    debug_dd        = 0
    debug_split     = 0
    debug_resolve   = 0
    ERR_GAIN =1
    Matirx_GAIN = 0    
    CE_DICT={}
    REL_UPD=set()
    LAST_MATRIX_SCORE=0

    def __init__(self):
        self.__resolving = 0
        self.__last_reported_length = 0
        self.monotony = 0
        self.outcome_cache  = OutcomeCache()
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
    def test(self, c,dest_dir,uid):
        """Test the configuration C.  Return PASS, FAIL, or UNRESOLVED"""
        c.sort(key=str)


        # If we had this test before, return its result
        if self.cache_outcomes:
            cached_result = self.outcome_cache.lookup(c)
            if cached_result != None:
                return cached_result


        # if self.monotony:
        #     # Check whether we had a passing superset of this test before
        #     cached_result = self.outcome_cache.lookup_superset(c)
        #     if cached_result == self.PASS:
        #         return self.PASS
            
            # cached_result = self.outcome_cache.lookup_subset(c)
            # if cached_result == self.FAIL:
            #     return self.FAIL


        if self.debug_test:
            logger.debug("test({})...".format(self.coerce(c)))

        outcome = self._test(c,dest_dir,uid)


        if self.debug_test:
            logger.debug("test({}) = {}".format(self.coerce(c), repr(outcome)))


        if self.cache_outcomes:
            self.outcome_cache.add(c, outcome)


        return outcome


    def _test(self,dest_dir,uid,keep_variant):
        """Stub to overload in subclasses"""
        return self.UNRESOLVED      # Placeholder
    
    def _build(self,c,uid=None, ignore_ref=False, keep_variant=False):
        """Stub to overload in subclasses"""
        return False,None,None,None    # Placeholder
 

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
            #print(title + ": " + repr(len(c)) + " deltas left:", self.coerce(c))
            self.__last_reported_length = len(c)




    def test_mix(self, csub, c, direction):
        if self.minimize:
            (t, csub) = self.test_and_resolve(csub, [], c, direction)
            if t == self.FAIL:
                return (t, csub)


        if self.maximize:
            csubbar = self.__listminus(self.CC, csub)
            cbar    = self.__listminus(self.CC, c)
            if direction == self.ADD:
                directionbar = self.REMOVE
            else:
                directionbar = self.ADD


            (tbar, csubbar) = self.test_and_resolve(csubbar, [], cbar,
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
        matrix = self._get_dep_matrix()
        
        self.set_tokens2hunks(cids=c)
        
        print(tabulate(matrix, tablefmt="fancy_grid", showindex=True, headers=list(range(len(c)))))    
        outcome = self.reldd(c,matrix=matrix)
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

    def sample(self,p):
        delset = []
        idx = np.argsort(p) # sort by probabilities and return index
        k = 0
        tmp = 1
        last = -99999
        idxlist = list(idx)
        i = 0
        while i < len(p):
            if p[idxlist[i]] == 0 :
                k = k + 1
                i = i + 1
                continue
            if not p[idxlist[i]] < 1 :
                break
            for j in range(k,i+1):
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
        return all(x >= 1 or x <= 0 for x in  p)

    def sample_x(self,p:list,retIdx:list, test_count_map:dict):
        kapa = random.randint(1, 11)
        if kapa < 4:
            sliced_map = {key: test_count_map[key] for key in retIdx}
            sorted_keys = [k for k, v in sorted(sliced_map.items(), key=lambda x: x[1])]
            idx = random.randint(1,len(retIdx)-1)
            idx2test = sorted_keys[:idx]
            delIdx = self.getIdx2test(retIdx, idx2test)
            print("select by test_count_map") 
        else:
            delIdx = self.sample(p)
            idx2test = self.getIdx2test(retIdx, delIdx) 
        return idx2test,delIdx
    
    def put_test_count(self, idx2test, test_count_map):      
        for num in idx2test:
            if num in test_count_map:
                test_count_map[num] += 1
            else:
                test_count_map[num] = 1


    def reldd(self, c, matrix: ndarray):
        print("Use RelDD")
        # assert self.test([]) == self.PASS #check wether F meet T
        retIdx = c[:]
        p = []
        his = set()
        hisCE = {}
        test_count_map={}
        for idx in range(0, len(c)):  # initialize the probability for each element in the input sequence
            p.append(0.1)
            test_count_map[idx] = 0
        last_p = []
        cc = 0
        i =0
        iscompile = True
        last_test=retIdx
        is_e =True
        sm = self.is_matrix_binary(matrix)
        random.seed(42)
        while not self.testDone(p):
            last_p = p[:]
            idx2test,delIdx = self.sample_x(p,retIdx,test_count_map)  
            delIdx = self.getIdx2test(retIdx, idx2test)
            addIdx=[]
            iscompile, dest_dir, uid,build_result = self.check_compile(idx2test,None,None,retIdx,matrix)           
            try:
                if not iscompile and not self.is_matrix_binary(matrix):                
                    iscompile, n_testIdx,addIdx,dest_dir, uid = self.predict_vaild_Idx(idx2test=idx2test,
                                                                                retIdx=retIdx, matrix=matrix, cpro=p,
                                                                                histotal=hisCE, build_info=build_result)              
                    if not iscompile:                    
                        dest_dir =None
                        uid = None
                    if iscompile:
                        if addIdx == None or len(addIdx) == 0 :
                            delIdx = self.getIdx2test(retIdx,idx2test) 
                            idx2test = n_testIdx

            except Exception as e:
                print(e)
                dest_dir =None
                uid = None
            
            self.LAST_MATRIX_SCORE = self.caculate_matrix_score(matrix) 
            print(f"当前增益{self.LAST_MATRIX_SCORE}")                                                
            res = self.test(idx2test, dest_dir, uid)
            self.put_test_count(idx2test=idx2test,test_count_map=test_count_map)
            his.add(self.get_list_str(idx2test))
            if  res == self.FAIL:  # set probabilities of the deleted elements to 0
                for set0 in range(0, len(p)):
                     if set0 not in idx2test:
                        p[set0] = 0
                        for index in range(0, len(matrix)):
                            matrix[index][set0] = 0.0
                            matrix[set0][index] = 0.0
                if len(idx2test) < len(retIdx):
                    self.reset(p,matrix)            
                retIdx = idx2test
            else:  # test(seq2test, *test_args) == PASS:
                p_cp = p[:]
                for setd in range(0, len(p)):
                    if setd in delIdx and 0 < p[setd] < 1:
                        delta = (self.computRatio(delIdx, p_cp) - 1) * p_cp[setd]
                        p[setd] = p_cp[setd] + delta       
            if set(last_p) == set(p):
                cc += 1
            if cc > 100:
                break
            print('{}:{}'.format(idx2test, res))
            print(p)
            print(tabulate(matrix, tablefmt="fancy_grid", showindex=True, headers=list(range(len(p)))))
            if (not sm) and self.is_matrix_binary(matrix):
                self.reset(p,matrix)
                self.REL_UPD.clear()
                sm =True
        print("{}->{}->{}".format(len(his), len(self.CE_DICT), retIdx))
        return retIdx
    
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
        print("update {}->{}".format(testIdx,delIdx))
        for itemt in testIdx:
            for itemd in delIdx:
                if matrix[itemt][itemd] != 0:
                    matrix[itemt][itemd] = min(matrix[itemt][itemd] / (1.0 - tmplog), 1.0)
                    if matrix[itemt][itemd] >= 0.99:
                        matrix[itemt][itemd] = 1.0
    
    def is_matrix_binary(self,m):
        arr = np.array(m)
        return np.all((arr == 0) | (arr == 1))

    def getCESubsets(self, testset: list, cemap: dict):
        res = []
        teststr = self.get_list_str(testset)
        for key, value in cemap.items():
            if key in teststr:
                res.append(value[0])
        return res
    
    def reset(self,p,matrix):
        for i in range(len(p)):
            if p[i]!=0:
                p[i] = 0.1
        self.resetMatrix(matrix)

        
    def resetMatrix(self,matrix):
        return None      
                
    def predict_vaild_Idx(self, idx2test, retIdx, matrix: ndarray, cpro: list, histotal: dict,build_info):
        print(f"{idx2test} 编译失败，尝试预测子集")

        n_testIdx,addIdx = self.resolve_dependency(idx2test=idx2test,retIdx=retIdx,matrix=matrix,cpro=cpro)
        is_compile, dtest_dir, uid,last_buildInfo = self.check_compile(n_testIdx, idx2test,build_info,retIdx,matrix) 
        if is_compile :
            return  is_compile,n_testIdx,addIdx,dtest_dir, uid     
        
        is_compile =False
        if self.ERR_GAIN == 1:
            matrix_score = self.check_matrix_score(matrix);
            # 首先根据log来快速降维关系
            is_compile,n_testIdx,addIdx,dtest_dir, uid = self.gen_subset_by_err(matrix,idx2test[:],retIdx,build_info)
            if self.check_matrix_score(matrix) == matrix_score: #无增益，说明log的操作无效
                self.ERR_GAIN =0
        if not is_compile:
        #如果增益为0， 则之后使用权重算法
            is_compile,n_testIdx,addIdx,dtest_dir, uid = self.gen_subset_by_matrix(matrix,idx2test[:],retIdx,build_info)  
        return  is_compile,n_testIdx,addIdx,dtest_dir, uid 
    
    def check_matrix_score(self,matrix):
        num_zeros = np.count_nonzero(matrix == 0)
        return num_zeros
    
    def check_compile(self,n_testIdx,last_testIdx,last_buildInfo,retIdx,matrix):
        err_key = self.get_list_str(n_testIdx)        
        if err_key in self.CE_DICT and self.CE_DICT[err_key]!=None:
            self.updateMatrix(n_testIdx,self.getIdx2test(retIdx,n_testIdx),matrix)
            return False, None,None,self.CE_DICT[self.get_list_str(n_testIdx)][1]
        if self.outcome_cache.lookup(n_testIdx)!=None:
            return True,None,None,None
        is_compile, dtest_dir, uid,build_info = self._build(n_testIdx)
        if not is_compile:
            self.CE_DICT[err_key] =(n_testIdx,build_info)
        #根据编译结果根新概率矩阵
        if (not is_compile):
            if build_info and last_buildInfo and len(build_info.err_set) < len(last_buildInfo.err_set) and last_testIdx:
                #更新概率
                if(n_testIdx < last_testIdx):
                    self.updateMatrix(n_testIdx,self.getIdx2test(last_testIdx,n_testIdx),matrix)
            else:
                self.updateMatrix(n_testIdx,self.getIdx2test(retIdx,n_testIdx),matrix)
                         
        if is_compile: # testIdx 和 delIdx之间的关系被解除
            for item in n_testIdx:
                for itemj in self.getIdx2test(retIdx,n_testIdx):
                    matrix[item][itemj] = 0
                               
        return  is_compile, dtest_dir, uid,build_info          
                       
    def gen_subset_by_err(self,matrix,testIdx,retIdx,build_info):
        #这个时候我们希望compile error占主导
        #1. 首先尝试添加元素
        print("通过log进行添加")
        is_compile =False
        last_testIdx = testIdx[:]
        last_buildInfo = copy.deepcopy(build_info)
        while not is_compile: 
            n_testIdx = list(set(last_testIdx) |set(last_buildInfo.rcids))
            if len(n_testIdx) == len(last_testIdx) or  len(n_testIdx) == len(retIdx): #不在能获取新的ADD,或者全集了结束
                break
            is_compile, dtest_dir, uid,last_buildInfo = self.check_compile(n_testIdx, last_testIdx,last_buildInfo,retIdx,matrix)            
            last_testIdx = n_testIdx
            if is_compile:
                addIdx = self.getIdx2test(n_testIdx,testIdx)
                return  is_compile,n_testIdx,addIdx, dtest_dir, uid       
        print("通过log进行删除")
        #2. 尝试删除元素
        last_testIdx = testIdx[:]
        last_buildInfo = copy.deepcopy(build_info)
        addIdx=[]
        while not is_compile:
            n_testIdx = list(set(last_testIdx)-set(last_buildInfo.rcids))
            if len(n_testIdx) == len(last_testIdx) or  len(n_testIdx) == len(retIdx) or len(n_testIdx) == 0: #不在能获取新的ADD,或者全集了结束
                break
            is_compile, dtest_dir, uid,last_buildInfo = self.check_compile(n_testIdx, last_testIdx,last_buildInfo,retIdx,matrix)            
            last_testIdx = n_testIdx
            if is_compile:
                return  is_compile,n_testIdx,addIdx, dtest_dir, uid   
        return False,addIdx,None,None,None
    
    
    def gen_subset_by_matrix(self,matrix,testIdx,retIdx,build_info):
        print("通过matrix进行添加")
        is_compile =False
        #首先尝试添加元素
        last_buildInfo = copy.deepcopy(build_info)
        his=set()
        delIdx = self.getIdx2test(retIdx,testIdx)
        i =0
        while not is_compile:
            if len(his)> min(3000,len(his)>2**len(delIdx)):
                break
            addIdx = self.gen_ADDset_from_matrix(matrix,testIdx,retIdx,last_buildInfo.rcids)
            n_testIdx = list(set(testIdx) | set(addIdx))
            if n_testIdx == None:
                break
            is_compile, dtest_dir, uid,last_buildInfo = self.check_compile(n_testIdx, testIdx,last_buildInfo,retIdx,matrix)
            
            now_score = self.caculate_matrix_score(matrix)
            if now_score == self.LAST_MATRIX_SCORE:
                i+=1
            else:
                self.LAST_MATRIX_SCORE =now_score
            if i> 100:
               break          
            if is_compile:
                return  is_compile,n_testIdx,addIdx,dtest_dir, uid
        
        print("通过matrix进行减少")         
        last_buildInfo = copy.deepcopy(build_info)
        his=set() 
        last_m_score =0.000  
        i =0   
        addIdx=[]
        while not is_compile:    
            if len(his)> min(3000,len(his)>2**len(delIdx)):
                break
            n_testIdx = self.gen_DELset_from_matrix(matrix,testIdx,retIdx,last_buildInfo.rcids)
            now_score = self.caculate_matrix_score(matrix)
            if now_score == self.LAST_MATRIX_SCORE:
                i+=1
            else:
                self.LAST_MATRIX_SCORE =now_score
            if i> 100:
               break  
            if len(n_testIdx) == len(retIdx) or len(n_testIdx) == 0:
                break
            is_compile, dtest_dir, uid,last_buildInfo = self.check_compile(n_testIdx, testIdx,last_buildInfo,retIdx,matrix)
            if is_compile: 
                return  is_compile, n_testIdx,addIdx,dtest_dir, uid     
        return False,None,addIdx,None,None
    
    def caculate_matrix_score(self,matrix1):
        nz1 = np.count_nonzero(matrix1)
        sum1 = np.sum(matrix1)
        return sum1/nz1
        
    def gen_DELset_from_matrix(self,matrix,testIdx,retIdx,err_cids):
        #权重算法，根据matrix中的依赖概率和err信息共同选择
        #1. 首先计算testIdx对外部元素的依赖程度
        #权重计算方式为matrix中testId一行的最大值
        delIdx = self.getIdx2test(retIdx,testIdx)
        sub_matirx = matrix[np.ix_(testIdx, delIdx)]
        test_del_weights =  [np.max(sub_matirx[testIdx.index(row),:]) for row in testIdx]
        #2. 将err提供的相关元素来融合权重 
        #融合方式为 wi+(1-wi)/2 *xi 其中xi=1表示err中的元素
        test_del_weights = [test_del_weights[index]+(1-test_del_weights[index])/2 if testIdx[index] in err_cids else test_del_weights[index] for index in range(len(testIdx))]
        
        #3.根据融合的概率权重进行随机选择
        del_idx =[]
        
        del_idx = self.weight_select(test_del_weights,testIdx)

        return list(set(testIdx) - set(del_idx))
        
    def gen_ADDset_from_matrix(self,matrix,testIdx,retIdx,err_cids):
        delIdx = self.getIdx2test(retIdx,testIdx)
        #权重算法，根据matrix中的依赖概率和err信息共同选择
        #1. 首先计算 delIdx中的每一个元素在matrix被依赖的权重
        #权重的计算方法为matrix中delId的一列的最大值
        sub_matirx = matrix[np.ix_(testIdx, delIdx)]
        del_add_weights = [np.max(sub_matirx[:,delIdx.index(column)]) for column in delIdx]

        #2. 将err提供的相关元素来融合权重 
        #融合方式为 wj+(1-wj)/2 *xi 其中xi=1表示err中的元素
        del_add_weights = [ del_add_weights[index]+(1-del_add_weights[index])/2 if delIdx[index] in err_cids else del_add_weights[index] for index in range(len(del_add_weights))]
        
        # 这里不对del_weights 进行归一化，原因是，每一个delIdx中元素被依赖的概率是独立的。不需要delIdx中的权重概率和是1

        #3.根据融合的概率权重进行随机选择
        add_idx =[]
        if all(item == 0 for item in del_add_weights):
            return None

        add_idx = self.weight_select(del_add_weights,delIdx)

        return list(set(add_idx))
       
    def weight_select(self,weights,delIdx):
        result=[]
        th = np.random.rand()
        for index in range(len(weights)):
            if weights[index] == 0:
                continue            
            u = np.random.rand()
            if u < weights[index]:
                result.append(delIdx[index])
        return result
     
    def get_del_by_random(self,weights,testIdx,cids):
        result=[]
        max_prob = max(weights)
        th = np.random.rand()
        for index in range(len(weights)):
            if testIdx[index] in cids:
                w = weights[index]
                weights[index] = w +(1-w)/2
                u = np.random.rand()
                if u < weights[index] / max_prob:
                    result.append(testIdx[index])
        print(f"random del{result}")
        return result         
        
    def get_dep_by_union(self,weights,delIdx,add_dep_from_log):

        result=[]    
        for index in range(len(weights)):
            if weights[index] == 0 and weights[index] in add_dep_from_log:
                add_dep_from_log.remove(weights[index])
        result =[delIdx[index] for index in self.sample(weights)]       
        return list((set(delIdx)-set(result)) | set(add_dep_from_log))
                    
    def get_list_str(self, t: list):
        if len(t) == 0:
            return ''
        t= list(t)
        t.sort()
        return ''.join(str(i)+'+' for i in t)       

    def add_dependency_Idx_back(self, idx2test, delIdx, matrix: ndarray):
        result = idx2test[:]
        tmp =[]
        for item in idx2test:
            del_depidx = self.sample(matrix[item,:].tolist())
            tmp = list(set(tmp) |set(del_depidx) )
        tmp = list(set(delIdx)-set(tmp))   
        result.extend(tmp)
        return result
    
    def remove_test_Idx(self, idx2test, matrix,retIdx,cpro,cids):
        n_testIdx = self.remove_testidx_by_ce_message(idx2test,cids)
        if(len(n_testIdx) == len(idx2test)):
            n_testIdx = self.remove_test_Idx_by_matrix(idx2test,matrix=matrix,cids=cids)
        n_testIdx = self.resolve_dependency(idx2test=n_testIdx,retIdx=retIdx,matrix=matrix,cpro=cpro)
        return n_testIdx
        
    def remove_test_Idx_by_matrix(self, idx2test, matrix, cids):
        idx2test = list(set(idx2test))
        prolist = []
        if len(idx2test) == 1:
            return []
        for item in idx2test:
            prolist.append(self.weighted_mean_row(matrix,row_idx=item))        
        result =[idx2test[index] for index in self.sample(prolist)]
        return list(set(result)-set(cids));
    
    def weighted_mean_column(self,matrix, column_idx):
         # 获取指定列的数据
        colunm = matrix[:, column_idx]
        # 找到非零元素的索引，并获取其对应的值
        nonzero_indices = np.nonzero(colunm)[0]
        values = colunm[nonzero_indices]
        # 计算非零元素的加权平均值
        sum_values= np.sum(values)
        count = len(nonzero_indices)
        if count == 0:
            return 0
        else:
            return sum_values / count
        
    def weighted_mean_row(self,matrix, row_idx):
        row = matrix[row_idx, :]
        nonzero_indices = np.nonzero(row)[0]
        values = row[nonzero_indices]
        # squares = np.square(values)
        sum_squares = np.sum(values)
        count = len(nonzero_indices)
        if count == 0:
            return 0
        else:
            return sum_squares / count
    
    def add_dep_by_ce_message(self,idx2test: list, cids: list):
        return list(set(idx2test)|set(cids))
    
    def remove_testidx_by_ce_message(self,idx2test: list, cids: list):
        return list(set(idx2test)-set(cids))    
    
    def conditional_entropy_gain(self,last_score,matrix):
        num_zeros = np.count_nonzero(matrix == 0)
        num_ones = np.count_nonzero(matrix == 1)
        score = (num_zeros + num_ones) / matrix.size
        return last_score -score, score
        
    def resolve_dependency(self, idx2test: list, retIdx: list, matrix, cpro: list):
        cp = idx2test[:]
        add=[]
        for item in cp:
            for i in range(0, len(cpro)):
                if matrix[item][i] == 1:
                    if (i not in cp) and (i not in add) and (i in retIdx):
                        add.append(i)
                    cpro[i] = cpro[item]
        #ADD
        while True:
            old_add =add[:]
            for item in add:
                for i in range(0, len(cpro)):
                    if matrix[item][i] == 1:
                        if (i not in add) and (i not in cp) and (i in retIdx):
                            add.append(i)
                        cpro[i] = cpro[item]
            add = list(set(add))            
            for item in add:
                for i in range(0, len(cpro)):
                    if matrix[item][i] == 1:
                        if (i not in cp) and (i not in cp) and (i in retIdx):
                            add.append(i)
                        cpro[i] = cpro[item]
            add = list(set(add))             
            if len(set(add)) == len(set(add)):
                break                       
        #Remove                        
        if len(set(cp))+len(set(add)) == len(set(retIdx)) or len(set(add)) == 0 :
            add=[]
            print("delete resolve")
            while True:
                old_result =cp[:]
                for item in cp:
                    for i in range(0, len(cpro)):
                        if matrix[item][i] == 1:
                            if i not in idx2test and item in cp :
                                cp.remove(item)
                cp = list(set(cp))            
                for item in cp:
                    for i in range(0, len(cpro)):
                        if matrix[item][i] == 1:
                            if i not in idx2test and item in cp:
                                cp.remove(item)
                cp = list(set(cp))                                            
                if len(set(cp)) == len(set(old_result)):
                    break             
         
        return list(cp),list(add)
    
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


            c_failed    = 0
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


    def ddmin(self, c):
        return self.ddgen(c, 1, 0)


    def ddmax(self, c):
        return self.ddgen(c, 0, 1)


    def ddmix(self, c):
        return self.ddgen(c, 1, 1)



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
                    progress    = 1
                    next_c2     = csub
                    next_n      = 2
                    cbar_offset = 0


                    if self.debug_dd:
                        logger.debug("dd: reduce c2 to {} deltas:".format(len(next_c2)))
                        logger.debug(self.pretty(next_c2))
                    break


                if t == self.PASS and t2 == self.FAIL:
                    # Reduce to complement
                    progress    = 1
                    next_c1     = csub
                    next_n      = max(next_n - 1, 2)
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
                    progress    = 1
                    next_c1     = csub
                    next_n      = 2
                    cbar_offset = 0


                    if self.debug_dd:
                        logger.debug("dd: increase c1 to {} deltas:".format(len(next_c1)))
                        logger.debug(self.pretty(next_c1))
                    break


                if t == self.FAIL and t1 == self.PASS:
                    # Increase
                    progress    = 1
                    next_c2     = csub
                    next_n      = max(next_n - 1, 2)
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


            c1  = next_c1
            c2  = next_c2
            n   = next_n
            run = run + 1


    def dd(self, c):
        return self.dddiff(c)           # Backwards compatibility


            




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
    (c, c1, c2) = mydd.dd([1, 2, 3, 4, 5, 6, 7, 8]) # Invoke DD
    logger.info("The 1-minimal failure-inducing difference is {}".format(c))
    logger.info("{} passes, {} fails".format(c1, c2))
    


# Local Variables:
# mode: python
# End: