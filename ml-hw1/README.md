# Time Series Classification

Two datasets are used: PEMS-SF and UWaveGestureLibrary, containing examples of time series recorded by multiple sensors/on multiple axes for road traffic on different days/8 different hand gestures. The classification problem is to find the week day/the gesture for new examples.

The first step taken is to visualize data in order to better understand potential important features. Next, we extract features and use selectors to discard features with not enough information.

This first work uses classic algorithms in order to classify correctly: SVC, Random Forest, Gradient Boosted Trees.
