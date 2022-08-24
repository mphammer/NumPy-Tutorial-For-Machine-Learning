# Numpy Tutorial for Machine Learning

This tutorial will teach you the basic skills needed to use NumPy for Machine Learning. It will provide a brief overview of NumPy's core features and then focus more specifically on aspects that are useful for Machine Learning problems.    

If you want to get the most out of this tutorial then I suggest cloning this repo and playing with the code as you go along.  

**What is NumPy?**   
"NumPy is the fundamental package for scientific computing in Python. It is a Python library that provides a multidimensional array object, various derived objects (such as masked arrays and matrices), and an assortment of routines for fast operations on arrays..." - [NumPy.org](https://numpy.org/doc/stable/user/quickstart.html)

**How does NumPy Relate to Machine Learning?**  
NumPy is the foundation for many Python libraries for machine learning. For example, [Pandas](https://pandas.pydata.org/docs/index.html), [Scikit-Learn](https://scikit-learn.org/stable/), and [matplotlib](https://matplotlib.org/) all use it. Functions from these libraries will often return NumPy's multidimensional array object or require them as input.      

NumPy's multidimensional array will commonly represent your Machine Learning Dataset, pieces of your Dataset, or ouput from methods like predicting values or calculating accuracy scores. Therefore, the most important things to know are how to look at values, do basic operations, and create/read/update/delete rows and columns from a two-dimensional array (ie a Table of Data).   

**Extra Learning**  
The guides on [numpy.org](https://numpy.org/doc/stable/user/quickstart.html) are very good and largely what this guide is based on.    

# Table of Contents

1. [Set Up Your Environment and Import NumPy](#set-up-your-environment-and-import-numpy)
2. [Introducing the Homogeneous Multidimensional Array](#introducing-the-homogeneous-multiimensional-array)
    1. [Key Array Features: Multidimensional and Homogeneous](#key-array-features-multidimensional-and-homogenous)
    2. [Quick Look at How Arrays are Printed](#quick-look-at-how-arrays-are-printed)
    3. [Basic Array Operations](#basic-array-operations)
    4. [Basic Array Methods](#basic-array-methods)
3. [Indexing Arrays](#indexing-arrays)
4. [Slicing Arrays](#slicing-arrays)
5. [Changing the Shape of Arrays](#changing-the-shape-of-arrays)
6. [CRUD Operations on Two Dimensional Arrays](#crud-create-read-update-delete-operations-on-two-dimensional-arrays-tables)
    1. [Create](#create)
    2. [Read](#read)
    3. [Update](#update)
    4. [Delete](#delete)

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

## Introducing the Homogeneous Multiimensional array

The homogenous n-dimensional array (narray) is the core of NumPy. 

The main features:
- Homogeneous - all its values are of one type
- N-dimensinoal (multidimensional)
- Extra Operations - it has more methods/operations than Python's built in array
- Fixed size - adding elements creates an entire new array
- Fast - operations are much faster than Python's built in array (list)

Note: Each dimension is called an "axis". So a 3-dimensional array has 3 axis.

### Key Array Features: Multidimensional and Homogenous


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



```python
# The Dot Product is Useful When Doing Linear Algebra 

np_array_a = np.array([1,2])
np_array_b = np.array([3,4])
print("The dot product of {} and {} is {}.".format(np_array_a, np_array_b, np_array_a.dot(np_array_b)))
```

    The dot product of [1 2] and [3 4] is 11.


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
print(one_dimensional_array)
print("one_dimensional_array[0] = {}".format(one_dimensional_array[0]))
print("one_dimensional_array[1] = {}".format(one_dimensional_array[1]))
print()

print("Dimensions: {}; Shape: {}".format(two_dimensional_array.ndim, two_dimensional_array.shape))
print(two_dimensional_array)
print("two_dimensional_array[0,0] = {}".format(two_dimensional_array[0,0]))
print("two_dimensional_array[1,3] = {}".format(two_dimensional_array[1,2]))
print("two_dimensional_array[0] = {}".format(two_dimensional_array[0]))
print()

print("Dimensions: {}; Shape: {}".format(three_dimensional_array.ndim, three_dimensional_array.shape))
print(three_dimensional_array)
print("three_dimensional_array[0,0] = {}".format(three_dimensional_array[0,0]))
print("three_dimensional_array[0,1] = {}".format(three_dimensional_array[0,1]))
print("three_dimensional_array[1,0] = {}".format(three_dimensional_array[1,0]))
print("three_dimensional_array[1,1] = {}".format(three_dimensional_array[1,1]))
print()
```

    Dimensions: 1; Shape: (3,)
    [1 2 3]
    one_dimensional_array[0] = 1
    one_dimensional_array[1] = 2
    
    Dimensions: 2; Shape: (2, 3)
    [[1 2 3]
     [4 5 6]]
    two_dimensional_array[0,0] = 1
    two_dimensional_array[1,3] = 6
    two_dimensional_array[0] = [1 2 3]
    
    Dimensions: 3; Shape: (2, 2, 4)
    [[[ 1  2  3  4]
      [ 5  6  7  8]]
    
     [[ 9 10 11 12]
      [13 14 15 16]]]
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
The slice is entered in the format `start_index:end_index`.     
For example, `np_array[0:2,0:2]` takes the first two elements in the first axis and from each of those it takes the first two elements. 


```python
''' 
NOTE: If you are using VS Code and the output is being truncated you need to: 
1. Open Settings and Search for "notebook.output.textLineLimit" 
2. Set that value to 1000
3. Close this file and re open it (don't just restart VS Code - actually re-open the file)
'''

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
print(one_dimensional_array)
print("one_dimensional_array[1:] = {}".format(one_dimensional_array[1:]))
print("one_dimensional_array[1:4] = {}".format(one_dimensional_array[1:4]))
print("one_dimensional_array[:] = {}".format(one_dimensional_array[:]))
print()

print("Dimensions: {}; Shape: {}".format(two_dimensional_array.ndim, two_dimensional_array.shape))
print(two_dimensional_array)
print("two_dimensional_array[:,0] = {} # Get all elements in the first dimension and take the 0th index (1st column)".format(two_dimensional_array[:,0]))
print("two_dimensional_array[:,1] = {} # Get all elements in the first dimension and take the 1st index (2nd column)".format(two_dimensional_array[:,0]))
print("two_dimensional_array[0,:] = {} # Get the first element in the first dimension and take all elements (1st row)".format(two_dimensional_array[0,:]))
print("two_dimensional_array[0:2,0:2] = \n{}".format(two_dimensional_array[0:2,0:2]))
print()

print("Dimensions: {}; Shape: {}".format(three_dimensional_array.ndim, three_dimensional_array.shape))
print(three_dimensional_array)
print("three_dimensional_array[0,0:2,0:2] = \n{}".format(three_dimensional_array[0,0:2,0:2]))
print("three_dimensional_array[1,0:2,0:2] = \n{}".format(three_dimensional_array[1,0:2,0:2]))
print("three_dimensional_array[:,0:2,0:2] = \n{}".format(three_dimensional_array[:,0:2,0:2]))
print()
```

    Dimensions: 1; Shape: (6,)
    [1 2 3 4 5 6]
    one_dimensional_array[1:] = [2 3 4 5 6]
    one_dimensional_array[1:4] = [2 3 4]
    one_dimensional_array[:] = [1 2 3 4 5 6]
    
    Dimensions: 2; Shape: (2, 6)
    [[ 1  2  3  4  5  6]
     [ 7  8  9 10 11 12]]
    two_dimensional_array[:,0] = [1 7] # Get all elements in the first dimension and take the 0th index (1st column)
    two_dimensional_array[:,1] = [1 7] # Get all elements in the first dimension and take the 1st index (2nd column)
    two_dimensional_array[0,:] = [1 2 3 4 5 6] # Get the first element in the first dimension and take all elements (1st row)
    two_dimensional_array[0:2,0:2] = 
    [[1 2]
     [7 8]]
    
    Dimensions: 3; Shape: (2, 2, 6)
    [[[ 1  2  3  4  5  6]
      [ 7  8  9 10 11 12]]
    
     [[13 14 15 16 17 18]
      [19 20 21 22 23 24]]]
    three_dimensional_array[0,0:2,0:2] = 
    [[1 2]
     [7 8]]
    three_dimensional_array[1,0:2,0:2] = 
    [[13 14]
     [19 20]]
    three_dimensional_array[:,0:2,0:2] = 
    [[[ 1  2]
      [ 7  8]]
    
     [[13 14]
      [19 20]]]
    


## Changing the Shape of Arrays


```python
# In Place - Modfies the array itself
print(".resize() modified the array in-place")
np_array = np.array([1,2,3,4,5,6])
print("Orignial Array Shape: {}".format(np_array.shape))
np_array.resize(3,2)
print("Original Array Shape: {} # see that it was modified".format(np_array.shape))
print()

# Out of Place - Returns a new array
print(".reshape() modified the array out-of-place")
np_array = np.array([1,2,3,4,5,6])
print("Original Array Shape: {}".format(np_array.shape))
new_np_array = np_array.reshape(3,2)
print("Original Array Shape: {} # see that it was NOT modified".format(np_array.shape))
print("New Array Shape: {}".format(new_np_array.shape))
```

    .resize() modified the array in-place
    Orignial Array Shape: (6,)
    Original Array Shape: (3, 2) # see that it was modified
    
    .reshape() modified the array out-of-place
    Original Array Shape: (6,)
    Original Array Shape: (6,) # see that it was NOT modified
    New Array Shape: (3, 2)


The size of the new array must be the same as the original array.   

It's easy to see how this function works when you have a one-dimensional list and then reshape it. It splits the data into the number of groups of the first axis, then it splits those into the number of the second axis, and then those into the same number of the third axis, etc.     

It's harder when you start with an array that's not one-dimensional. The best way to think about it is that the function first flattens the array (check out the ravel function) and then reshapes it.     


```python
# Example with an one-dimensional array
original_array = np.array([1,2,3,4,5,6])
print("Orignial Array: \n{}".format(original_array))
print("Raveled Original Array: \n{}".format(original_array.ravel()))
print()
print("original_array.reshape(1,6): \n{}".format(original_array.reshape(1,6)))
print("original_array.reshape(6,1): \n{}".format(original_array.reshape(6,1)))
print("original_array.reshape(3,2): \n{}".format(original_array.reshape(3,2)))
print("original_array.reshape(2,3): \n{}".format(original_array.reshape(2,3)))
print("original_array.reshape(3,2,1): \n{}".format(original_array.reshape(3,2,1)))
print("original_array.reshape(2,3,1): \n{}".format(original_array.reshape(2,3,1)))
print("original_array.reshape(1,3,2): \n{}".format(original_array.reshape(1,3,2)))
print("original_array.reshape(1,2,3): \n{}".format(original_array.reshape(1,2,3)))
```

    Orignial Array: 
    [1 2 3 4 5 6]
    Raveled Original Array: 
    [1 2 3 4 5 6]
    
    original_array.reshape(1,6): 
    [[1 2 3 4 5 6]]
    original_array.reshape(6,1): 
    [[1]
     [2]
     [3]
     [4]
     [5]
     [6]]
    original_array.reshape(3,2): 
    [[1 2]
     [3 4]
     [5 6]]
    original_array.reshape(2,3): 
    [[1 2 3]
     [4 5 6]]
    original_array.reshape(3,2,1): 
    [[[1]
      [2]]
    
     [[3]
      [4]]
    
     [[5]
      [6]]]
    original_array.reshape(2,3,1): 
    [[[1]
      [2]
      [3]]
    
     [[4]
      [5]
      [6]]]
    original_array.reshape(1,3,2): 
    [[[1 2]
      [3 4]
      [5 6]]]
    original_array.reshape(1,2,3): 
    [[[1 2 3]
      [4 5 6]]]



```python
original_array = np.array([[1,2,3],[4,5,6]])
print("Orignial Array: \n{}".format(original_array))
print("Raveled Original Array: \n{}".format(original_array.ravel()))
print()
print("original_array.reshape(1,6): \n{}".format(original_array.reshape(1,6)))
print("original_array.reshape(3,2): \n{}".format(original_array.reshape(3,2)))
print("original_array.reshape(2,3,1): \n{}".format(original_array.reshape(2,3,1)))
print("original_array.reshape(1,2,3): \n{}".format(original_array.reshape(1,2,3)))
```

    Orignial Array: 
    [[1 2 3]
     [4 5 6]]
    Raveled Original Array: 
    [1 2 3 4 5 6]
    
    original_array.reshape(1,6): 
    [[1 2 3 4 5 6]]
    original_array.reshape(3,2): 
    [[1 2]
     [3 4]
     [5 6]]
    original_array.reshape(2,3,1): 
    [[[1]
      [2]
      [3]]
    
     [[4]
      [5]
      [6]]]
    original_array.reshape(1,2,3): 
    [[[1 2 3]
      [4 5 6]]]



```python
# Transpose is a useful operation for Linear Algebra

np_array = np.array([
        [1,2,3],
        [4,5,6],
        [7,8,9]
    ])
print("Original Array:\n{}".format(np_array))
print("Transposed Array:\n{}".format(np.transpose(np_array)))
```

    Original Array:
    [[1 2 3]
     [4 5 6]
     [7 8 9]]
    Transposed Array:
    [[1 4 7]
     [2 5 8]
     [3 6 9]]


## CRUD (Create, Read, Update, Delete) Operations on Two Dimensional Arrays (Tables)

- Create: Add new columns and rows      
- Read: Get/print columns, rows, and specific indexes      
- Update: Update columns, rows, and specific indexes      
- Delete: Remove columns, rows, and specific indexes      

Your data will often look like a table which is essentially a 2 dimensional array. These are some of the basic operations that you will want to be able to do on a table.    

### Create

#### Create Columns


```python
# Hstack takes an array with the same number of axis and same number of columns, and concatenates to the right
# H Stack = Horizontal Stack 
# https://numpy.org/doc/stable/reference/generated/numpy.hstack.html

my_data = np.array(
    [
        [ 1, 2, 3],
        [ 5, 6, 7]
    ]
)
print("Initial 2D Array (Table):\n{}".format(my_data))

new_column_values = np.array([4,8])
new_column_reshaped = new_column_values.reshape(2,1)
print("New Column:\n{}".format(new_column_reshaped))

updated_my_data = np.hstack((my_data, new_column_reshaped))
print("Updated Data Table:\n{}".format(updated_my_data))
```

    Initial 2D Array (Table):
    [[1 2 3]
     [5 6 7]]
    New Column:
    [[4]
     [8]]
    Updated Data Table:
    [[1 2 3 4]
     [5 6 7 8]]



```python
# column_stack

column_a = np.array([ 1, 4, 7,10])
column_b = np.array([ 2, 5, 8,11])
column_c = np.array([ 3, 6, 9,12])

my_data = np.column_stack((column_a, column_b, column_c))

print("Updated Data Table:\n{}".format(my_data))
```

    Updated Data Table:
    [[ 1  2  3]
     [ 4  5  6]
     [ 7  8  9]
     [10 11 12]]



```python
# np.c_
# https://numpy.org/doc/stable/reference/generated/numpy.c_.html
column_a = np.array([ 1, 4, 7,10])
column_b = np.array([ 2, 5, 8,11])
column_c = np.array([ 3, 6, 9,12])

np.c_[column_a, column_b, column_c]
```




    array([[ 1,  2,  3],
           [ 4,  5,  6],
           [ 7,  8,  9],
           [10, 11, 12]])



#### Create Rows

The main ways to do this are by using the `np.vstack()` method and the `np.r_` method.    
Note: `np.row_stack()` is exactly the same as `np.vstack()` (This is different from the above `np.hstack()` and `np.column_stack()`)    


```python
# Vstack takes an array with the same number of axis and same number of rows, and concatenates to the bottom
# V Stack = Vertical Stack 
# https://numpy.org/doc/stable/reference/generated/numpy.vstack.html#numpy.vstack

my_data = np.array(
    [
        [ 1, 2, 3],
        [ 4, 5, 6]
    ]
)
print("Initial 2D Array (Table):\n{}".format(my_data))

new_column_values = np.array([7,8,9])
new_column_reshaped = new_column_values.reshape(1,3)
print("New Column:\n{}".format(new_column_reshaped))

updated_my_data = np.vstack((my_data, new_column_reshaped))
print("Updated Data Table:\n{}".format(updated_my_data))
```

    Initial 2D Array (Table):
    [[1 2 3]
     [4 5 6]]
    New Column:
    [[7 8 9]]
    Updated Data Table:
    [[1 2 3]
     [4 5 6]
     [7 8 9]]



```python
# np.r_ 
# Read more about this here: https://numpy.org/doc/stable/reference/generated/numpy.r_.html
row_a = np.array([1,2,3])
row_b = np.array([4,5,6])
row_c = np.array([7,8,9])

np.r_['0,2', row_a, row_b, row_c]
```




    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])



### Read

See the section on Indexing Arrays. Below are two examples for getting a specific row and a specific column.     


```python
two_dimensional_array = np.array(
    [[ 1, 2, 3], 
     [ 4, 5, 6],
     [ 7, 8, 9]]
)

def get_row(row_index, two_d_array):
    return two_d_array[row_index]

print("Row {}: {}".format(0, get_row(0, two_dimensional_array)))
print("Row {}: {}".format(1, get_row(1, two_dimensional_array)))
print("Row {}: {}".format(2, get_row(2, two_dimensional_array)))

def get_col(col_index, two_d_array):
    return two_d_array[:, col_index]

print("Col {}: {}".format(0, get_col(0, two_dimensional_array)))
print("Col {}: {}".format(1, get_col(1, two_dimensional_array)))
print("Col {}: {}".format(2, get_col(2, two_dimensional_array)))
```

    Row 0: [1 2 3]
    Row 1: [4 5 6]
    Row 2: [7 8 9]
    Col 0: [1 4 7]
    Col 1: [2 5 8]
    Col 2: [3 6 9]


### Update

The basic way to do this is to take a slice of the array and set that equal to your new values.  


```python
two_dimensional_array = np.array(
    [[ 1, 2, 3], 
     [ 4, 5, 6],
     [ 7, 8, 9]]
)
print("Initial Array:\n{}".format(two_dimensional_array))

two_dimensional_array[0,:] = [9, 9, 9]
print("Change Row 0:\n{}".format(two_dimensional_array))

two_dimensional_array[:,0] = [0, 0, 0]
print("Change Col 0:\n{}".format(two_dimensional_array))
```

    Initial Array:
    [[1 2 3]
     [4 5 6]
     [7 8 9]]
    Change Row 0:
    [[9 9 9]
     [4 5 6]
     [7 8 9]]
    Change Col 0:
    [[0 9 9]
     [0 5 6]
     [0 8 9]]


### Delete 

It's easiest to use the `np.delete()` method. Below I show how it can be done with slicing.    


```python
two_dimensional_array = np.array(
    [[ 1, 2, 3], 
     [ 4, 5, 6],
     [ 7, 8, 9]]
)

print("Original Array:\n{}".format(two_dimensional_array))

ROW_AXIS = 0
COLUMN_AXIS = 1

removed_row_array = np.delete(two_dimensional_array, 1, ROW_AXIS)
print("Remove Row Index 1:\n{}".format(removed_row_array))

removed_col_array = np.delete(two_dimensional_array, 1, COLUMN_AXIS)
print("Remove Col Index 1:\n{}".format(removed_col_array))
```

    Original Array:
    [[1 2 3]
     [4 5 6]
     [7 8 9]]
    Remove Row Index 1:
    [[1 2 3]
     [7 8 9]]
    Remove Col Index 1:
    [[1 3]
     [4 6]
     [7 9]]



```python
two_dimensional_array = np.array(
    [[ 1, 2, 3], 
     [ 4, 5, 6],
     [ 7, 8, 9]]
)

print("Original Array:\n{}".format(two_dimensional_array))

def remove_row(row_idx, two_d_array):
    top_table = two_d_array[0:row_idx]
    bottom_table = two_d_array[row_idx+1:]
    return np.vstack((top_table, bottom_table))

print("Remove Row 0:\n{}".format(remove_row(0, two_dimensional_array)))
print("Remove Row 1:\n{}".format(remove_row(1, two_dimensional_array)))
print("Remove Row 2:\n{}".format(remove_row(2, two_dimensional_array)))
```

    Original Array:
    [[1 2 3]
     [4 5 6]
     [7 8 9]]
    Remove Row 0:
    [[4 5 6]
     [7 8 9]]
    Remove Row 1:
    [[1 2 3]
     [7 8 9]]
    Remove Row 2:
    [[1 2 3]
     [4 5 6]]



```python
two_dimensional_array = np.array(
    [[ 1, 2, 3], 
     [ 4, 5, 6],
     [ 7, 8, 9]]
)

print("Original Array:\n{}".format(two_dimensional_array))

def remove_col(col_idx, two_d_array):
    left_table = two_d_array[:,0:col_idx]
    right_table = two_d_array[:,col_idx+1:]
    return np.hstack((left_table, right_table))

print("Remove Col 0:\n{}".format(remove_col(0, two_dimensional_array)))
print("Remove Col 1:\n{}".format(remove_col(1, two_dimensional_array)))
print("Remove Col 2:\n{}".format(remove_col(2, two_dimensional_array)))
```

    Original Array:
    [[1 2 3]
     [4 5 6]
     [7 8 9]]
    Remove Col 0:
    [[2 3]
     [5 6]
     [8 9]]
    Remove Col 1:
    [[1 3]
     [4 6]
     [7 9]]
    Remove Col 2:
    [[1 2]
     [4 5]
     [7 8]]

