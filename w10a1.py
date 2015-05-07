def npv(n, t, r):
    presentValue = n / float((1 + r / 100.0) ** t)
    
    return presentValue

flows1 = [-25, -25, -25, -25, -25, 20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20, 20]

flows2 = [-25, -25, -25, -25, -25, -25, 20, 20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20]
        
flows3 = [-25, -25, -25, -25, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20]

flows4 = [-25, -25, -25, -25, -25, 5, 10, 15, 20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20]

rs = [5, 7.8, 9, 11, 15]
NPV1, NPV2, NPV3, NPV4 = 0, 0, 0, 0

for r in rs:
    for i in range(len(flows4)):
        '''PV1 = npv(flows1[i], i + 1, r)
        NPV1 += PV1
        
        PV2 = npv(flows2[i], i + 1, r)
        NPV2 += PV2'''
        
        #PV3 = npv(flows3[i], i + 1, r)
        #NPV3 += PV3
        
        PV4 = npv(flows4[i], i + 1, r)
        NPV4 += PV4
    print "Scenario 4 NPV (r = %s): " % r + str(NPV4)
    #print "Scenario 3 NPV (r = %s): " % r + str(NPV3)
    NPV4 = 0

#print "Scenario 1 NPV: " + str(NPV1)
#print "Scenario 2 NPV: " + str(NPV2)
#print "Scenario 3 NPV: " + str(NPV3)
