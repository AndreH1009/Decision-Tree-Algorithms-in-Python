# Decision-Tree-Algorithms-in-Python
In this project I implement Decision-Trees using common algorithm schemes like the C4.5 scheme. It is possible to visualize a generated tree using visualize(), which generates LaTex code to create a graphical representation in "dtree.pdf". The current state of the project does not feature Pruning yet.

## Requirements
- LaTex incl. packages: forest, nicefrac, color
- Python 3.0 or later, incl. packages: pandas

## How To
- access the project:\
clone the project and find file "dtrees.py" in src.
- generate a Decision Tree on DataFrame 'df' using 'play' as the classification attribute:\
```tree = build_decision_tree(df, 'play')```
- create file "dtree.pdf" containing a graphical representation of the tree:\
```visualize_tree(tree)```
- classify a new datapoint 'x', which is a DataFrame, using Decision-Tree "tree":\
```tree.classify(x)```

## Example
![Screenshot](/docs/images/dtree.png)