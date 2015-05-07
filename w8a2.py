#############################
#   Justine De Caires       #
#   Formal Analyses 2.8.2   #
#   justine@minerva.kgi.edu #
#############################

import numpy.random as npr
import random

def matrix(students, sessions):
    """ Deterministically creates a schedule matrix, S = [students][sessions],
        such that s[i][j] = 1 if student i leads session j, and s[i][j] = 0
        otherwise. Each student leads exactly 2 sessions, each session is led
        by exactly 2 students and two students cannot lead more than one session
        together.
        
        Output matrix:
        [[1 1 0 0 0 0 0 0 0 0 0 0 0 0]
        [0 1 1 0 0 0 0 0 0 0 0 0 0 0]
        [0 0 1 1 0 0 0 0 0 0 0 0 0 0]
        [0 0 0 1 1 0 0 0 0 0 0 0 0 0]
        [0 0 0 0 1 1 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 1 1 0 0 0 0 0 0 0]
        [0 0 0 0 0 0 1 1 0 0 0 0 0 0]
        [0 0 0 0 0 0 0 1 1 0 0 0 0 0]
        [0 0 0 0 0 0 0 0 1 1 0 0 0 0]
        [0 0 0 0 0 0 0 0 0 1 1 0 0 0]
        [0 0 0 0 0 0 0 0 0 0 1 1 0 0]
        [0 0 0 0 0 0 0 0 0 0 0 1 1 0]
        [0 0 0 0 0 0 0 0 0 0 0 0 1 1]
        [1 0 0 0 0 0 0 0 0 0 0 0 0 1]] """
    
    # Initializing S to a students x sessions zero matrix
    S = npr.random_integers(0, 0, (students, sessions))
    i = 0
    j = 0
    
    # Iterating through students
    while i < students:
        
        # Constructing the output matrix
        for k in range(2):
            S[i][j] = 1
            j += 1
        
        if j == sessions:
            j = -1
        else:
            j -= 1
        i += 1
    
    return S

def matrix2(students, sessions):
    """ matrix(students, sessions) modified such that there are at least 3
        weeks between two sessions led by a given student. """
        
    # Initializing S to a students x sessions zero matrix
    S = npr.random_integers(0, 0, (students, sessions))
    
    i = 0
    j = 0
    
    # Iterating through students
    while i < students:
        
        # Constructing the output matrix
        S[i][j] = 1
        j += 4
        
        S[i][j] = 1
        j -= 5
        
        if j == sessions:
            j = -1
        i += 1
    
    return S

print "Without modification: \n", matrix(14, 14)
print
print "With modification (a): \n", matrix2(14, 14)