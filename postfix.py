from convert import Convert
pa = []
class Postfix:

    def __init__(self, capacity): 
        self.top = -1
        self.capacity = capacity 
        # This array is used a stack  
        self.array = [] 
        self.postfixarr = []
    #to covert the string to postfix expression
    def topostfix(self,string):
        obj = Convert(len(string))
        #print ("capacity ",self.capacity)
        self.postfixarr = obj.infixToPostfix(string)
        #print("postfix array is :",self.postfixarr)
        return self.postfixarr
    # check if the stack is empty 
    def isEmpty(self): 
        return True if self.top == -1 else False
      
    # Return the value of the top of the stack 
    def peek(self): 
        return self.array[-1] 
      
    # Pop the element from the stack 
    def pop(self): 
        if not self.isEmpty(): 
            self.top -= 1
            return self.array.pop() 
        else: 
            return "$"
      
    # Push the element to the stack 
    def push(self, op): 
        self.top += 1
        self.array.append(op)  
  
  
    # The main function that converts given infix expression 
    # to postfix expression 
    def evaluatePostfix(self, exp): 
        # Iterate over the expression for conversion 
        for i in exp: 
              
            # If the scanned character is an operand 
            # (number here) push it to the stack 
            if i.isdigit(): 
                self.push(i) 
            # If the scanned character is an operator, 
            # pop two elements from stack and apply it. 
            else: 
                val1 = self.pop() 
                val2 = self.pop() 
                self.push(str(eval(val2 + i + val1))) 
                #print(self.array[self.top])
        return (str(self.pop()))

class B:
    def __init__(self):   
        self.arr = []
    def callToPostfix(self,string):
        obj1 = Postfix(0)
        self.arr = obj1.topostfix(string)
        obj1 = Postfix(len(self.arr))
        return  obj1.evaluatePostfix(self.arr)