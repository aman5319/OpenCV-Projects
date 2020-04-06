import re
import numpy as np
from pprint import pprint

def parse(q1):
    """
    :param q1: a list or a tuple example [1,0,0,1,0] 
    This Function generates a list of parent child relationship
    example [parent1,child11,child111,child112]
    This means there is one parent it has one child and that child has two childs.
    """
    
    l=[]
    i=0
    parent=None
    level = []
    while i < len(q1):
    
        if i < len(q1) and q1[i]==0:
            k=0
            
            while i < len(q1) and abs(q1[i])==0:
                k+=1
                i+=1
            l.extend(["parent{0}".format(aa+1) for aa in range(k+1)])
        
        if i < len(q1) and q1[i]==1:
            i+=1
            if len(l)==0:
                l.extend(["parent{0}".format(i)])
            if parent is None:
                parent=i
            k=0
            
            while i < len(q1) and abs(q1[i])==0:
                k+=1
                i+=1
            l.extend(["child{0}{1}".format(parent , aa+1) for aa in range(k+1)])
            level.append(parent)
            parent = "{0}{1}".format(parent , aa+1)

            
        if i < len(q1) and q1[i]==-1:

            i+=1
            k=0
            while i< len(q1) and abs(q1[i])==0:
                k+=1
                i+=1
            if len(level)==1:
                
                le = int(level.pop())
                l.extend(["parent{0}".format(aa+1+le) for aa in range(k+1)])
                parent=aa+1+le
            elif len(level) >1:
                le = level.pop()
                le = int(le)
                l.extend(["child{0}".format(aa+1+le) for aa in range(k+1)])
                parent=aa+1+le
            elif len(level)==0 and len(l)==0:
                l.extend(["parent{0}".format(i)])

        
    return l
    
def generateGraph(q):
    """
    :param this takes a parent child relationship list 
    Generates a dictionary in graph data structure
    """
    dic={}
    parent=None
    child_level= None
    child=None
    for i  , cord in q:
        if i.find("parent")!=-1:
            dic[cord]={}
            parent=cord
            child_level=None
            child=None

        if i.find("child") != -1:
            c_level =int(i.strip("child"))
            if child_level is None or c_level-child_level==1:
                #print(child)
                if child is None:
                    dic[parent][cord]={}
                else:
                    dic[parent][child][cord]={}
            elif c_level-child_level<0:
                child=None
                dic[parent][cord]={}
            else:
                child="child{0}".format(child_level)
                child = dict(q)[child]
                dic[parent][child][cord ]={}
            child_level=c_level
    return dic

def sortCheckbox(d):
    """
    Takes a checkbox dictionary and returns a numpy array by combining two checkbox in a single array , denominator (y2-y1) , sorted dictionary
    """
    d = sorted(d,key= lambda x: (x[0][1], x[0][0]))
    l=[]
    for fi ,se in zip(d[:-1],d[1:]):
        l.append((fi[0][0] , fi[0][1],se[0][0] ,se[0][1]))

    c =np.asarray(l).astype(np.float)
    return c ,(c[:,3]-c[:,1]) ,d
    
def preprocess(d):
    """
    Takes a checkbox dictionary and returns a list containing 1,0,-1 
    0 means two checkbox in same alignment
    1 means parent child relationship
    -1 means child parent relationship 
    """
    c ,denom,d = sortCheckbox(d)
    
    for i in np.nonzero(denom<4)[0]:
        d[i][0][1],d[i+1][0][1] = d[i][0][1] ,d[i][0][1]
        
    c ,denom,d = sortCheckbox(d)
    denom = (c[:,3]-c[:,1])
    q  = (c[:,2]-c[:,0]) / denom
    q1=np.where(abs(q)<=0.1,0,q) 
    q1 = np.where(q1>0 , 1,q1)
    q1 = np.where(q1<0 , -1,q1)
    q1 = q1.astype(np.int)
    q1 = q1.tolist()
    d = tuple((tuple(i[0]) , i[1]) for i in d)
    return q1,d

def run(d):
    """
    combine all the methods and get a dictionary
    """
    q1 ,d= preprocess(d)#; pprint(d); pprint(q1)
    l=parse(q1)
    l = zip(l,d)
    l=generateGraph(l)#;pprint(l)
    return l

if __name__=="__main__":
    d =[[[419, 1558, 28, 25], 'unchecked'],
 [[415, 1414, 28, 25], 'unchecked'],
 [[415, 1359, 28, 25], 'checked'],
 [[466, 1138, 28, 25], 'unchecked'],
 [[466, 1055, 28, 25], 'unchecked'],
 [[466, 1000, 28, 24], 'unchecked'],
 [[416, 861, 28, 25], 'unchecked'],
 [[416, 779, 28, 25], 'unchecked'],
 [[417, 613, 29, 24], 'unchecked'],
 [[418, 557, 28, 25], 'unchecked'],
 [[370, 447, 28, 25], 'checked'],
 [[321, 255, 29, 25], 'unchecked'],
 [[411, 254, 28, 25], 'checked']]
    run(d)
