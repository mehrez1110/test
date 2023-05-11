cdef class Rect:
    def __cinit__(self,string inputText,int y1=0,int x1=0,int width=0,int height=0,float weight=0.0,int x2=0,int y2=0):
        splitInput = inputText.split(" ")
        
        self.x1=int(splitInput[0])-1
        self.y1=int(splitInput[1])-1
        
        
        
        self.width=int(splitInput[2])
        self.height=int(splitInput[3])
     
        self.weight=float(splitInput[4])    
        
        self.y2=self.y1+self.height
        self.x2=self.x1+self.width    
        

    cdef float calculateArea(self,window,scale):
        #self.printRect()
       # print("window values: ",window[self.y2][self.x2],window[self.y1][self.x1],window[self.y1][self.x2],window[self.y2][self.x1])
        float scaledX1=self.x1*scale
        float scaledY1=self.y1*scale
        float scaledY2=self.y2*scale
        float scaledX2=self.x2*scale
        float area=(window[scaledY2][scaledX2]+window[scaledY1][scaledX1]-window[scaledY1][scaledX2]-window[scaledY2][scaledX1])
       # print("weight*area/size: ",(self.weight*area)/(np.size(window)))
        #return (self.weight*area)/((window[0][0]+window[-1][-1]-window[0][-1]-window[-1][-0]))
        return (self.weight*area)/np.size(window)
    
    def printRect(self):
        print('Rect')
        print('x1: ',self.x1)
        print('x2: ',self.x2)
        print('y1: ',self.y1)
        print('y2: ',self.y2)
        print('width: ',self.width)
        print('height: ',self.height)
        print('weight: ',self.weight)