# ML_Module_02

## for executing file
```
python3 <filename>
```

## for checking format
```
pycodestyle <filename>
```

## for modifying format
```
black <filename>
```

## for copying docker file
```
docker cp <docker container id>:/app/results .
docker exec -it <CONTAINER_ID> ./test.sh
docker cp <CONTAINER_ID>:/app/results .
```

## conclusion
1. What is the main (obvious) difference between univariate and multivariate linear regression?
- The main difference is the number of input features. Univariate linear regression involves one input feature, while multivariate linear regression involves two or more input features.
2. Is there a minimum number of variables needed to perform a multivariate linear regression?
- 2
3. Is there a maximum number of variables needed to perform a multivariate linear regression? In theory and in practice?
- In theory No. In practive, if number is big, it tends to cause overfitting.
4. Is there a difference between univariate and multivariate linear regression in terms of performance evaluation?
- Same.
5. What does it mean geometrically to perform a multivariate gradient descent with two variables?
- Geometrically, performing gradient descent with two variables in multivariate linear regression can be visualized as finding the lowest point in a three-dimensional landscape.
6. Can you explain what is overfitting?
- It's like fitting the model too closely to the peculiarities of the training set, which hampers its ability to generalize.
7. Can you explain what is underfitting?
- Underfitting happens when a model is too simple to capture the underlying pattern in the data.
8. Why is it important to split the data set in a training and a test set?
- Splitting the data into training and test sets is crucial for evaluating the model's ability to generalize to new, unseen data. It helps in assessing the true predictive power of the model and in detecting overfitting.
9. If a model overfits, what will happen when you compare its performance on the training set and the test set?
- If a model overfits, it will typically perform very well on the training set but poorly on the test set.
10. If a model underfits, what do you think will happen when you compare its perfor- mance on the training set and the test set?
- If a model underfits, it will likely show poor performance on both the training and test sets. 