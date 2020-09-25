from .bases import BaseImageDataset
import os.path as osp
import os
from collections import defaultdict


class NAIC(BaseImageDataset):
    def __init__(self, root='../data', verbose = True):
        super(NAIC, self).__init__()
        self.dataset_dir = root
        self.dataset_dir_train = osp.join(self.dataset_dir, 'train')
        self.dataset_dir_test = osp.join(self.dataset_dir, 'test')
        # 以[(img_path, label, 1),...]格式存储训练数据路径和对应label
        train = self._process_dir(self.dataset_dir_train, relabel=True)
        # 以[(img_path, 1, 1),...]格式存储测试数据路径
        query_green, query_normal = self._process_dir_test(self.dataset_dir_test,  query = True)
        gallery_green, gallery_normal = self._process_dir_test(self.dataset_dir_test, query = False)


        if verbose:
            print("=> NAIC Competition data loaded")
            self.print_dataset_statistics(train, query_green+query_normal, gallery_green+gallery_normal)

        self.train = train
        self.query_green = query_green
        self.gallery_green = gallery_green
        self.query_normal = query_normal
        self.gallery_normal = gallery_normal

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)


    def _process_dir(self, data_dir, relabel=True):
        filename = osp.join(data_dir, 'label.txt')
        dataset = []
        camid = 1
        count_image=defaultdict(list)
        with open(filename, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()
                if not lines:
                    break

                # img_name,img_label = [i for i in lines.split()]
                img_name, img_label = lines.split(':')
                if img_name == 'train/105180993.png' or img_name=='train/829283568.png' or img_name=='train/943445997.png': # remove samples with wrong label
                    continue
                count_image[img_label].append(img_name)
        val_imgs = {}
        pid_container = set()
        for pid, img_name in count_image.items():
            if len(img_name) < 2:
                pass
            else:
                val_imgs[pid] = count_image[pid]
                pid_container.add(pid)
        # 按顺序，按顺序为每个ID分配从0开始的label
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        # 将每张图片与对应ID的label关联起来：[(img_path, label, 1),...]
        for pid, img_name in val_imgs.items():
            pid = pid2label[pid]
            for img in img_name:
                dataset.append((osp.join(data_dir,'images', img), pid, camid))

        return dataset


    # 需要修改
    def _process_dir_test(self, data_dir, query=True):
        if query:
            subfix = 'query'
        else:
            subfix = 'gallery'
        filename = osp.join(data_dir, subfix)
        dataset = []
        for img_name in os.listdir(filename):

            dataset.append((osp.join(self.dataset_dir_test, subfix, img_name), 1, 1))

        dataset_green = dataset
        return dataset_green, dataset

