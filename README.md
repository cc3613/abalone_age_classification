# abalone_age_classification
Using measurements of abalones to predict the age of such abalone, done in various methods

In this project, I tried using different methods (some from sklearn libraries) to perform the prediction. The key is to use a number of different measurements (ex. length, diameter, shell weights, etc.) to predict the age of an abalone. In the past, the age is determined by cutting the shell through the cone, staining it, and counting the number of rings through a microscope -- a boring and time-consuming task. However, since there are other measurements done on different attributes, being able to predict the age accurately using these numbers can reduce the amount of labor on this task.

The dataset is acquired from UCI Machine Learning Repository

Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

There are total of 8 attributes and 1 tag (no. rings, divided into 8 groups based on 28 different ages seen). I'm doing it as a classification problem.

First experiment was to use all attributes provided to check the accuracy. Now this is not the best approach, but I like to do it as a starting point to get an idea of what it would look like using all information given. The first method was a K-NNR method with k=15. I've used k ranging from 3 to 15 and found that there's not much difference for any k, so I kept the last value.

I've written my own K-NNR and used sklearn's K-NNR for comparison, and it turned out that there's not much difference between these two, as expected. The accuracy for my own 15-NNR algo is 23.7%, while sklearn's 15-NNR has a 24.8% for accuracy. This result is not satisfying, yet expected, as curse of dimensionality will catch on this easily. On top of that, having a 28-group classification is itself a hard problem. The problem is that my algorithm is way slower than sklearn's. For the similar accuracies, my algorithm's performance needs some improvement.

I next used a random forest algorithm with the same number of attributes. The result is similar at 23.7%, with 100 estimators. Intuitively, a random forest algorithm would be more fitted to this question as some attributes are categorical, namely they are discrete. When mixing discrete and continuous numbers together, it would be better using some methods that are suited for these types of situations.

Another ensemble algorithm I tried was gradient boosted regressor, again from the sklearn library. The result shows that this algorithm seems to be better for this type of problem. With an accuracy of 54.2%, it more than doubled the accuracy from 15-NNR and RF. It seems like GBR is smarter when dealing with different types of attributes. In real world, data is not always clean (very rare, in fact), so having something that can deal with diffrent types of attributes saves a lot of time and labor of preprocessing (it can potentially avoid the errors done in preprocessing, as well)

knowing that this many classes can make the problem really hard, I reduced the number of classes by combining some classes together (ex. age 1-4 in group 0, age 5-8 in group 1, and so on) and created 7 classes from the original 28. I then re-ran the algorithms to see the new results. My own 15-NNR jumped from 23.7% to 66.7%, almost 3 times. Sklearn's 15-NNR also jumped to 64.9%, keep in mind this algorithm is still much faster. The random forest method produced a 67% accuracy, also a big jump. One thing to note is that the accuracy for GBR algo for 7 classes actually dropped, on average, though it's still pretty close to the original 28-class result. The explanation here is that it could be because the GBR's algorithm is already smart enough that it can distinguish the differences among 28 classes, and mixing the classes together actually confuses the algorithm, thus producing a poorer result.

Now that the number of classes are reduced, maybe the number of attributes can be reduced as well? This can save the computational time should the result be on par with the raw data.

I've used principle component analysis to find the most important vectors as attributes, dropping the attributes to only 3 principle components.

I'll use a table to present the numbers:


<a href='http://imgur.com/QiN20pY'> Table: Accuracy in % for different algorithms and processing.</a>

From the table, it seems PCA does not change the result much, so it would be a good idea to implement it if processing time is not a problem (O(n^3)). On the other hand, if storage is not a concern, it is okay to go ahead and use the original data.

A quick plot of confusion matrix (Random Forest, 7-class, pca) shows that most of the errors are happening between group 1 and group 2, and group 2 and group 3.

<a href='http://imgur.com/ahChfpQ'>confusion matrix for 7-class abalone ring count</a>

On a side note, using the GB regressor instead of classifier will return a ~50% for all cases, but that'll be another study and won't be mentioned too much here.

