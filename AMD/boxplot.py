from matplotlib import pyplot as plt


accuracies = [0.4444444444444444, 0.625, 1.0, 0.875, 1.0]
auc_scores = [0.516875, 0.6677777777777778, 1.0, 0.9234693877551021, 0.9500000000000001]

plt.boxplot([accuracies, auc_scores],
            notch=True,
            vert=True, 
            patch_artist=True)
plt.ylim(0, 1)
plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0])
plt.xticks([1,2],['accuracy','macro AUC score'])
plt.savefig('boxplot.png')