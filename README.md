# Numpy Tutorial for Machine Learning

This tutorial will teach you the basic skills needed to use NumPy for Machine Learning. It will provide a brief overview of NumPy's core features and then focus more specifically on aspects that are useful for Machine Learning problems.  

**What is NumPy?**   
"NumPy is the fundamental package for scientific computing in Python. It is a Python library that provides a multidimensional array object, various derived objects (such as masked arrays and matrices), and an assortment of routines for fast operations on arrays..." - [NumPy.org](https://numpy.org/doc/stable/user/quickstart.html)

**How does NumPy Relate to Machine Learning?**  
As the definition states, "NumPy is the fundamental package for scientific computing in Python". This means that popular Python libraries for machine learning are built on top of it (For example, [Pandas](https://pandas.pydata.org/docs/index.html), [Scikit-Learn](https://scikit-learn.org/stable/), and [matplotlib](https://matplotlib.org/)). It's common to use NumPy's n-dimensional array when using the other libraries or for the other libraries to have similar functionality. 

The fundamental building block of NumPy is the multidimensinal array. The NumPy array can show up in different ways during a Machine Learning problem but it will commonly represent your Dataset, pieces of your Dataset, or ouput from methods like predicting values or calculating accuracy scores. Therefore, the most important things to know are how to look at values, create/read/update/delete rows (examples), and create/read/update/delete columns (features). 

**Extra Learning**  
The guides on [numpy.org](https://numpy.org/doc/stable/user/quickstart.html) are very good and largely what this guide is based on.  

## Set Up Your Environment and Import NumPy

Create a Virtual Environment (For Mac)  
```
$ python -m venv venv  
$ source venv/bin/activate  
```

Install numpy  
```$ pip install numpy  ```


```python
import numpy as np
```

## Introducing the Homogeneous N-Dimensional array

The homogenous n-dimensional array (narray) is the core of NumPy. 

The main features:
- Homogeneous - all its values are of one type
- N-dimensinoal (multidimensional)
- Extra Operations - it has more methods/operations than Python's built in array
- Fixed size - adding elements creates an entire new array
- Fast - operations are much faster than Python's built in array (list)

Note: Each dimension is called an "axis". So a 3-dimensional array has 3 axis.

### A Look At How Arrays Are Multi-Dimensional and Homogenous


```python
print("Standard Python Array (list):")
one_dimensional_python_array = [1,2,3,4]
print(one_dimensional_python_array)
print(type(one_dimensional_python_array))
print()

print("A One Dimensional NumPy array that we created from the Python array:")
one_dimensional_numpy_array = np.array([1,2,3,4])
print(one_dimensional_numpy_array)
print(type(one_dimensional_numpy_array))
```

    Standard Python Array (list):
    [1, 2, 3, 4]
    <class 'list'>
    
    A One Dimensional NumPy array that we created from the Python array:
    [1 2 3 4]
    <class 'numpy.ndarray'>



```python
print("Python arrays (lists) aren't really multidimensional. They do not have buult in methods for finding their shape or dimensions")
two_dimensional_python_array = [[1,2,3,4], [5,6,7,8]]
rows = len(two_dimensional_python_array)
cols = len(two_dimensional_python_array[0])
print(two_dimensional_python_array)
print("Shape: {}".format((rows, cols)))
print("Size: {}".format(rows*cols))
dimensions = 0
temp_array = two_dimensional_python_array
while type(temp_array) == list:
    dimensions += 1
    temp_array = temp_array[0]
print("Dimensions: {}".format(dimensions))
print()

print("NumPy Array:")
two_dimensional_numpy_array = np.array([[1,2,3,4], [5,6,7,8]])
print(two_dimensional_numpy_array)
print("Shape: {}".format(two_dimensional_numpy_array.shape))
print("Size: {}".format(two_dimensional_numpy_array.size))
print("Dimensions: {}".format(two_dimensional_numpy_array.ndim))
```

    Python arrays (lists) aren't really multidimensional. They do not have buult in methods for finding their shape or dimensions
    [[1, 2, 3, 4], [5, 6, 7, 8]]
    Shape: (2, 4)
    Size: 8
    Dimensions: 2
    
    NumPy Array:
    [[1 2 3 4]
     [5 6 7 8]]
    Shape: (2, 4)
    Size: 8
    Dimensions: 2



```python
print("Three Dimensional Array:")
three_dimensional_python_array_1 = np.array(
    [[[1,2,3,4]],
    [[5,6,7,8]]]
)
print(three_dimensional_python_array_1)
print("Shape: {}".format(three_dimensional_python_array_1.shape))
print("Size: {}".format(three_dimensional_python_array_1.size))
print("Dimensions: {}".format(three_dimensional_python_array_1.ndim))
print()

print("Three Dimensional Array:")
three_dimensional_python_array_2 = np.array(
    [[[1,2],[3,4]],
    [[5,6],[7,8]]]
)
print(three_dimensional_python_array_2)
print("Shape: {}".format(three_dimensional_python_array_2.shape))
print("Size: {}".format(three_dimensional_python_array_2.size))
print("Dimensions: {}".format(three_dimensional_python_array_2.ndim))
```

    Three Dimensional Array:
    [[[1 2 3 4]]
    
     [[5 6 7 8]]]
    Shape: (2, 1, 4)
    Size: 8
    Dimensions: 3
    
    Three Dimensional Array:
    [[[1 2]
      [3 4]]
    
     [[5 6]
      [7 8]]]
    Shape: (2, 2, 2)
    Size: 8
    Dimensions: 3



```python
print("Four Dimensional Array")
four_dimensional_python_array_1 = np.array(
    [[[[1,2,3,4]],
    [[5,6,7,8]]]]
)
print(four_dimensional_python_array_1)
print("Shape: {}".format(four_dimensional_python_array_1.shape))
print("Size: {}".format(four_dimensional_python_array_1.size))
print("Dimensions: {}".format(four_dimensional_python_array_1.ndim))
print()

print("Four Dimensional Array")
four_dimensional_python_array_2 = np.array(
    [[[[1],[2]], [[3],[4]]],[[[5],[6]], [[7],[8]]]]
)
print(four_dimensional_python_array_2)
print("Shape: {}".format(four_dimensional_python_array_2.shape))
print("Size: {}".format(four_dimensional_python_array_2.size))
print("Dimensions: {}".format(four_dimensional_python_array_2.ndim))
```

    Four Dimensional Array
    [[[[1 2 3 4]]
    
      [[5 6 7 8]]]]
    Shape: (1, 2, 1, 4)
    Size: 8
    Dimensions: 4
    
    Four Dimensional Array
    [[[[1]
       [2]]
    
      [[3]
       [4]]]
    
    
     [[[5]
       [6]]
    
      [[7]
       [8]]]]
    Shape: (2, 2, 2, 1)
    Size: 8
    Dimensions: 4


#### Quick Look At How Arrays Are Printed

- The last dimension (axis) is printed horizontally.  
- The 2nd to last dimension (axis) is printed vertically.  
- The other dimensions (axis) are separated by empty lines. And there is an extra line for each additional dimension.  

**One Dimensional**  
There is only a "last dimension" so it's just printed horizontally. 
```
[0 1 2 3]
```

**Two Dimensions**  
Each item in the second to last dimension are printed on a new row (vertically).  
And the last dimension is printed horixontally.  
```
[[1 2 3]
 [4 5 6]]
```

**Three Dimensions**  
The items in the third dimension are separated by a single space.  
Notice how it looks like two "two dimensional" arrays are separated by an empty line.  
```
[[[ 1  2  3]
  [ 4  5  6]]
  
 [[ 7  8  9]
  [10 22 12]]]
```

**Four Dimensions**
The items in the fourth dimension are separated by two spaces.  
The items in the third dimension are still separated by a single space.  
```
[[[[ 0  1  2]
   [ 3  4  5]]

  [[ 6  7  8]
   [ 9 10 11]]]


 [[[12 13 14]
   [15 16 17]]

  [[18 19 20]
   [21 22 23]]]]
```

### Basic Array Operations


```python
# Elementwise arithmetic operators
a = np.array([1,2,3])
b = np.array([3,2,1])
print(a-b)  # [(1-3) (2-2) (3-1)]
print(a+b)  # [(1+3) (2+2) (3+1)]
print(a*2)  # [(1*2) (2*2) (3*2)]
print(b**2) # [(3^2) (2^2) (1^2)]
```

    [-2  0  2]
    [4 4 4]
    [2 4 6]
    [9 4 1]


### Basic Array Methods


```python
np_array = np.array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
print("The Sum of All Elements is: {}".format(np_array.sum()))
print("The Sum of Each Column is: {}".format(np_array.sum(axis=0)))
print("The Sum of Each Row is: {}".format(np_array.sum(axis=1)))
print("The Min Elements is: {}".format(np_array.min()))
print("The Min of Each Column: {}".format(np_array.min(axis=0)))
print("The Max Elements is: {}".format(np_array.max()))
print("The Max of Each Row: {}".format(np_array.max(axis=1)))
```

    The Sum of All Elements is: 66
    The Sum of Each Column is: [12 15 18 21]
    The Sum of Each Row is: [ 6 22 38]
    The Min Elements is: 0
    The Min of Each Column: [0 1 2 3]
    The Max Elements is: 11
    The Max of Each Row: [ 3  7 11]


## Indexing Arrays  
For each axis (dimension) there can be 1 index.
Looking at the `.shape` attribute is a useful way to know what the range of values is for each dimension.  


```python
one_dimensional_array = np.array(
    [1,2,3]
)
two_dimensional_array = np.array(
    [[1,2,3], 
    [4,5,6]]
)
three_dimensional_array = np.array(
    [
        [[ 1, 2, 3, 4],
         [ 5, 6, 7, 8]],

        [[ 9,10,11,12],
         [13,14,15,16]],
    ]
)
print("Dimensions: {}; Shape: {}".format(one_dimensional_array.ndim, one_dimensional_array.shape))
print("one_dimensional_array[0] = {}".format(one_dimensional_array[0]))
print("one_dimensional_array[1] = {}".format(one_dimensional_array[1]))
print()

print("Dimensions: {}; Shape: {}".format(two_dimensional_array.ndim, two_dimensional_array.shape))
print("two_dimensional_array[0,0] = {}".format(two_dimensional_array[0,0]))
print("two_dimensional_array[1,3] = {}".format(two_dimensional_array[1,2]))
print("two_dimensional_array[0] = {}".format(two_dimensional_array[0]))
print()

print("Dimensions: {}; Shape: {}".format(three_dimensional_array.ndim, three_dimensional_array.shape))
print("three_dimensional_array[0,0] = {}".format(three_dimensional_array[0,0]))
print("three_dimensional_array[0,1] = {}".format(three_dimensional_array[0,1]))
print("three_dimensional_array[1,0] = {}".format(three_dimensional_array[1,0]))
print("three_dimensional_array[1,1] = {}".format(three_dimensional_array[1,1]))
print()
```

    Dimensions: 1; Shape: (3,)
    one_dimensional_array[0] = 1
    one_dimensional_array[1] = 2
    
    Dimensions: 2; Shape: (2, 3)
    two_dimensional_array[0,0] = 1
    two_dimensional_array[1,3] = 6
    two_dimensional_array[0] = [1 2 3]
    
    Dimensions: 3; Shape: (2, 2, 4)
    three_dimensional_array[0,0] = [1 2 3 4]
    three_dimensional_array[0,1] = [5 6 7 8]
    three_dimensional_array[1,0] = [ 9 10 11 12]
    three_dimensional_array[1,1] = [13 14 15 16]
    



```python
print("Use the Shape of the NumPy Array to print each element:")
def print_array(np_array):
    # Base Case: Print each element if it's the last demension (aka it's a 1D array)
    if np_array.ndim == 1:
        for element in np_array:
            print("{} ".format(element), end="")
        return 
    # Loop over each sub-np_array at this dimension
    shape = np_array.shape
    num_np_arrays = shape[0]
    for i in range(num_np_arrays):
        print_array(np_array[i])
        print()

print("Shape: {}".format(three_dimensional_array.shape))
print_array(three_dimensional_array)

```

    Use the Shape of the NumPy Array to print each element:
    Shape: (2, 2, 4)
    1 2 3 4 
    5 6 7 8 
    
    9 10 11 12 
    13 14 15 16 
    


## Slicing Arrays

For each axis (dimension) there can be 1 slice. Again, it's useful to look at the `.shape` attribute.  


```python
one_dimensional_array = np.array(
    [1,2,3,4,5,6]
)
two_dimensional_array = np.array(
    [[ 1, 2, 3, 4, 5, 6], 
     [ 7, 8, 9,10,11,12]]
)
three_dimensional_array = np.array(
    [
        [[ 1, 2, 3, 4, 5, 6],
         [ 7, 8, 9,10,11,12]],

        [[13,14,15,16,17,18],
         [19,20,21,22,23,24]],
    ]
)
print("Dimensions: {}; Shape: {}".format(one_dimensional_array.ndim, one_dimensional_array.shape))
print("one_dimensional_array[1:] = {}".format(one_dimensional_array[1:]))
print("one_dimensional_array[1:4] = {}".format(one_dimensional_array[1:4]))
print("one_dimensional_array[:] = {}".format(one_dimensional_array[:]))
print()

print("Dimensions: {}; Shape: {}".format(two_dimensional_array.ndim, two_dimensional_array.shape))
print("two_dimensional_array[:,0] = {} # Get all elements in the first dimension and take the 0th index (1st column)".format(two_dimensional_array[:,0]))
print("two_dimensional_array[:,1] = {} # Get all elements in the first dimension and take the 1st index (2nd column)".format(two_dimensional_array[:,0]))
print("two_dimensional_array[0,:] = {} # Get the first element in the first dimension and take all elements (1st row)".format(two_dimensional_array[0,:]))
print("two_dimensional_array[0:2,0:2] = {}".format(two_dimensional_array[0:2,0:2]))
print()

print("Dimensions: {}; Shape: {}".format(three_dimensional_array.ndim, three_dimensional_array.shape))
print("three_dimensional_array[0,0:2,0:2] = {}".format(three_dimensional_array[0,0:2,0:2]))
print("three_dimensional_array[1,0:2,0:2] = {}".format(three_dimensional_array[1,0:2,0:2]))
print()
```

    Dimensions: 1; Shape: (6,)
    one_dimensional_array[1:] = [2 3 4 5 6]
    one_dimensional_array[1:4] = [2 3 4]
    one_dimensional_array[:] = [1 2 3 4 5 6]
    
    Dimensions: 2; Shape: (2, 6)
    two_dimensional_array[:,0] = [1 7] # Get all elements in the first dimension and take the 0th index (1st column)
    two_dimensional_array[:,1] = [1 7] # Get all elements in the first dimension and take the 1st index (2nd column)
    two_dimensional_array[0,:] = [1 2 3 4 5 6] # Get the first element in the first dimension and take all elements (1st row)
    two_dimensional_array[0:2,0:2] = [[1 2]
     [7 8]]
    
    Dimensions: 3; Shape: (2, 2, 6)
    three_dimensional_array[0,0:2,0:2] = [[1 2]
     [7 8]]
    three_dimensional_array[1,0:2,0:2] = [[13 14]
     [19 20]]
    

