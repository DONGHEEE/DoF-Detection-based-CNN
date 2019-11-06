import os
import glob
import shutil

savepath = '/Users/jhkimMultiGpus/PycharmProjects/tf_Tutorial/SRCNN-Tensorflow-master/Test/focus_image'
savepath_input = '/Users/jhkimMultiGpus/PycharmProjects/tf_Tutorial/SRCNN-Tensorflow-master/Test/focus_label'

def prepare_data(dataset):
    if dataset != "Test":
        data_dir = os.path.join(os.getcwd(), dataset)
        image = glob.glob(os.path.join(data_dir, "*.jpg"))
        for item in image:
            itemq = str(item.split("\\")[-1])
            itemq = os.path.splitext(itemq)[0]
            itemq = str(itemq)
            if item.find("_grey") is -1:
                if item.find("_dof") is not -1:
                    #if int(itemq[-5]) % 2 == 0:
                    print(item)
                    shutil.copy2(item, savepath_input)
                else:
                    #if int(itemq[-1]) % 2 == 0:
                    print(item)
                    shutil.copy2(item, savepath)

data = prepare_data(dataset="Train/DoF_Images (2)")
