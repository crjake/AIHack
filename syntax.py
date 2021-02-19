import numpy as np 
import matplotlib.pyplot as plt
import scipy as sp

[print("Hello \n") for i in range(10) if i%2==0]

x = [1,2,3,4]
y = np.array([1,2,3,4])
sq2 = y**2

sq = [i**2 for i in x]

#for int i = 1; i < something; i = i + 3 {
#}

np.linspace(0, 10, 51)

for i in [[1,2], [3,4]]:
    print(i)
    print(i[0])

#class_instance.radius



for i in range(1,10):
    if i%5==0:
        break
    else:
        print(i)


       
class Class:
    x = 0
    def __init__(self, radius):
        """ start here
        """
        self.radius = radius
        self._hidden = 10
        self.__mangled = 15

        Class.x += 1
        

    def volume(self):
        """Calculates the volume of object
        
        Paramaters:
            radius = float, radius of object

        Return
            Volume = float, volume of object
        """
        return 4/3 * np.pi * self.radius**3

        some_variable_name
        
        
        


def subClass(Class):
    def __init__(self,radius,somethingElse):
        super(subClass, self).__init__(radius)
        self.somethingElse = somethingElse


