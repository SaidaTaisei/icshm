import matplotlib.pyplot as plt

class Helper:
    @staticmethod
    def visualize(test_x,test_y,predict_y,index):
        i=index
        fig, axes = plt.subplots(2, 8)
        axes[0,0].imshow(test_x[i])
        axes[0,1].imshow(test_y[i][:, :, 0])
        axes[0,2].imshow(test_y[i][:, :, 1])
        axes[0,3].imshow(test_y[i][:, :, 2])
        axes[0,4].imshow(test_y[i][:, :, 3])
        axes[0,5].imshow(test_y[i][:, :, 4])
        axes[0,6].imshow(test_y[i][:, :, 5])
        axes[0,7].imshow(test_y[i][:, :, 6])
        axes[1,0].imshow(test_x[i])
        axes[1,1].imshow(predict_y[i][:, :, 0])
        axes[1,2].imshow(predict_y[i][:, :, 1])
        axes[1,3].imshow(predict_y[i][:, :, 2])
        axes[1,4].imshow(predict_y[i][:, :, 3])
        axes[1,5].imshow(predict_y[i][:, :, 4])
        axes[1,6].imshow(predict_y[i][:, :, 5])
        axes[1,7].imshow(predict_y[i][:, :, 6])
        for i in range(axes.shape[0]):
            for j in range(axes.shape[1]):
                axes[i,j].axis("off")
        plt.show()
    @staticmethod
    def saveImage(test_x,test_y,predict_y):
        plt.clf()
        plt.close()
        for i in range(predict_y.shape[0]):
            fig, axes = plt.subplots(2, 8, figsize=(6, 2))
            axes[0, 0].imshow(test_x[i])
            axes[0, 1].imshow(test_y[i][:, :, 0])
            axes[0, 2].imshow(test_y[i][:, :, 1])
            axes[0, 3].imshow(test_y[i][:, :, 2])
            axes[0, 4].imshow(test_y[i][:, :, 3])
            axes[0, 5].imshow(test_y[i][:, :, 4])
            axes[0, 6].imshow(test_y[i][:, :, 5])
            axes[0, 7].imshow(test_y[i][:, :, 6])
            axes[1, 0].imshow(test_x[i])
            axes[1, 1].imshow(predict_y[i][:, :, 0])
            axes[1, 2].imshow(predict_y[i][:, :, 1])
            axes[1, 3].imshow(predict_y[i][:, :, 2])
            axes[1, 4].imshow(predict_y[i][:, :, 3])
            axes[1, 5].imshow(predict_y[i][:, :, 4])
            axes[1, 6].imshow(predict_y[i][:, :, 5])
            axes[1, 7].imshow(predict_y[i][:, :, 6])
            for k in range(axes.shape[0]):
                for j in range(axes.shape[1]):
                    axes[k, j].axis("off")
            fig.savefig("result/"+str(i)+".png")
            plt.clf()
            plt.close()
