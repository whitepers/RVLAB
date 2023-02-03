import os
import Layers as layers
import GPUtil
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import metrics
# from sklearn import metrics
import tqdm

GPU = -1

if GPU == -1:
    devices = "%d" % GPUtil.getFirstAvailable(order="memory")[0]
else:
    devices = "%d" % GPU

os.environ["CUDA_VISIBLE_DEVICES"] = devices

l = keras.layers
K = keras.backend

# TODO: Set the mode
Mode = 0 #TODO: 모드만 바꾸면 (즉, 0이냐 1이냐) 학습하는 모델이 바뀌는 스위치 같은거야
file = "%s" % Mode
file_name = {'0': "Quantile_Gaussian", '1': "ResNet18_SE"}

ch = 64
fold = 5 # 데이터를 5개의 group으로 나누고 4개의 데이터를 훈련으로 사용하여 모델 훈련
epoch = 140
learning_rate = 0.0001
learning_decay = 0.98
batch_size = 3

class Utils:
    def __init__(self):
        self.data_path = '/Users/오용석/Desktop/ADNI-Dataset/' #TODO: 너가 저장한 데이터 path로 바꾸기

    def load_adni_data(self):
        """
        class #0: NC(433), class #1: pMCI(251), class #2: sMCI(497), class #3: AD(359)
        :return: NC, pMCI, sMCI, AD
        """
        dat = np.load(self.data_path + "total_dat.npy", mmap_mode="r")
        lbl = np.load(self.data_path + "labels.npy")

        return dat, lbl

    def data_permutation(self, lbl, cv): #TODO: 여기는 형이 NC랑 AD 이진분류 하려고 두 데이터만 label로부터 뽑아서 처리한거야 / 즉, 다른 종류로 classification 세팅하고 싶으면 당연히 바꾸면 돼
        Total_NC_idx, Total_AD_idx = np.squeeze(np.argwhere(lbl == 0)), np.squeeze(np.argwhere(lbl == 3))
        amount_NC, amount_AD = int(len(Total_NC_idx) / 5), int(len(Total_AD_idx) / 5)############# 왜 나누기 5를 해주는지

        NCvalid_idx = Total_NC_idx[cv * amount_NC:(cv + 1) * amount_NC]
        NCtrain_idx = np.setdiff1d(Total_NC_idx, NCvalid_idx)
        NCtest_idx = NCvalid_idx[:int(len(NCvalid_idx) / 2)]
        NCvalid_idx = np.setdiff1d(NCvalid_idx, NCtest_idx)

        ADvalid_idx = Total_AD_idx[cv * amount_AD:(cv + 1) * amount_AD]
        ADtrain_idx = np.setdiff1d(Total_AD_idx, ADvalid_idx)
        ADtest_idx = ADvalid_idx[:int(len(ADvalid_idx) / 2)]
        ADvalid_idx = np.setdiff1d(ADvalid_idx, ADtest_idx)

        self.train_all_idx = np.concatenate((NCtrain_idx, ADtrain_idx))
        self.valid_all_idx = np.concatenate((NCvalid_idx, ADvalid_idx))
        self.test_all_idx = np.concatenate((NCtest_idx, ADtest_idx))

        return self.train_all_idx, self.valid_all_idx, self.test_all_idx

    def seperate_data(self, data_idx, dat, lbl, CENTER=False):
        dat, lbl = dat[data_idx], lbl[data_idx]
        dat = np.squeeze(dat)
        lbl = np.where(lbl == 3, 1, lbl).astype("int32") #TODO: permutation이 바뀌면 여기 label 처리도 달라져야겠지
        lbl = np.eye(2)[lbl.squeeze()]
        lbl = lbl.astype('float32')
        if len(data_idx) == 1: dat = np.expand_dims(dat, axis=0)

        # Original
        if CENTER:
            for batch in range(len(data_idx)): #TODO: 여기는 데이터 전처리! (나는 quantile norm이랑 gaussian norm 사용했어)
                # Quantile normalization
                Q1, Q3 = np.quantile(dat[batch], 0.1), np.quantile(dat[batch], 0.9) # Quantile은 MRI 데이터 특성상 갑자기 pixel값이 튀는 outlier값들이 좀 있어서 그거 잡아주려고 쓰는거
                dat[batch] = np.where(dat[batch] < Q1, Q1, dat[batch])              # ex) 대체로 0~341 pixel값의 분포를 가지는 이미지인데 특정 영역에서 막 1000 이렇게 찍히면 그거 341로 맞춰주는거
                dat[batch] = np.where(dat[batch] > Q3, Q3, dat[batch])

                # Gaussian normalization
                m, std = np.mean(dat[batch]), np.std(dat[batch])
                dat[batch] = (dat[batch] - m) / std
            dat = np.expand_dims(dat, axis=-1)

        else: #TODO: 여기서 center가 아닌 else문은 data augmentation하려고 random rotation 해주느라 쓴거 (조건문을 두는 이유는 training할때만 augmentation을 사용해야하고 validation이나 test때는 꼭 (정규화된) 원본으만 사용해야해)
            padding = 5
            npad = ((padding, padding), (padding, padding), (padding, padding))
            emp = np.empty(shape=(dat.shape[0], dat.shape[1], dat.shape[2], dat.shape[3]))

            for cnt, dat in enumerate(dat):
                tmp = np.pad(dat, npad, "constant")
                emp[cnt] = tf.image.random_crop(tmp, emp[cnt].shape)

            for batch in range(len(emp)):
                # Quantile normalization
                Q1, Q3 = np.quantile(emp[batch], 0.1), np.quantile(emp[batch], 0.9)
                emp[batch] = np.where(emp[batch] < Q1, Q1, emp[batch])
                emp[batch] = np.where(emp[batch] > Q3, Q3, emp[batch])

                # Gaussian normalization
                m, std = np.mean(emp[batch]), np.std(emp[batch])
                emp[batch] = (emp[batch] - m) / std
            dat = np.expand_dims(emp, axis=-1)

        return dat, lbl

class ResNet18: # TODO: 여기서 SE들어가는 모듈들은 그냥 squeeze-and-excitation 적용한 모듈도 만들어본건데 일부러 안지웠어 한번 봐봐 즉, mode=0은 그냥 resnet18, mode=1은 SE 들어간 resnet18
    def __init__(self, ch=ch):
        self.ch = ch
        self.input = layers.input_layer(input_shape=(96, 114, 96, 1), name="input")
        self.build_model()

    def conv_bn_act(self, x, f, n, s=1, k=None, p="same", act=True, trans=False, out_p="auto"):
        if trans:
            c_layer = layers.conv_transpose
        else:
            c_layer = layers.conv

        if k:
            conv_l = c_layer(f=f, p=p, k=k, s=s, out_p=out_p, name=n + "_conv")
        else:
            conv_l = c_layer(f=f, p=p, name=n + "_conv")
        out = conv_l(x)

        norm_l = layers.batch_norm(name=n + "_norm")
        out = norm_l(out)

        if act:
            act_l = layers.relu(name=n + "_relu")
            out = act_l(out)
        return out

    def concat(self, x, y, n):
        concat_l = layers.concat(name=n + "_concat")
        return concat_l([x, y])

    def flatten_layer(self, x, n=None):
        flatten_l = layers.flatten(n + "_flatten")(x)
        return flatten_l

    def dense_layer(self, x, f, act=None, n=None):
        dense_l = layers.dense(f, act=None, name=n + "_dense")
        out = dense_l(x)

        if act:
            act_l = layers.relu(n + "_relu")
            out = act_l(out)
        return out

    def residual_block(self, x, ch, name):
        shortcut = x
        x = self.conv_bn_act(x=x, k=3, f=ch, s=1, p="same", act=True, n="enc_conv%d_1" % name)
        x = self.conv_bn_act(x=x, k=3, f=ch, s=1, p="same", act=False, n="enc_conv%d_2" % name)
        act = layers.relu(name="enc_conv%d_2_relu" % name)
        x = act(x + shortcut)
        return x

    def residual_SE_block(self, x, ch, ratio, name):
        shortcut = x
        ch_reduced = ch // ratio

        x = self.conv_bn_act(x=x, k=3, f=ch, s=1, p="same", act=True, n="enc_conv%d_1" % name)
        x = self.conv_bn_act(x=x, k=3, f=ch, s=1, p="same", act=False, n="enc_conv%d_2" % name)  # activation True or False

        squeeze = layers.global_avgpool(rank=3, name="GAP%d" % name)(x)
        excitation = layers.dense(f=ch_reduced, act="relu", b=False, name="excitation%d_1" % name)(squeeze)
        out = layers.dense(f=ch, act="sigmoid", b=False, name="excitation%d_2" % name)(excitation)

        out = tf.keras.layers.Reshape((1, 1, 1, ch), name="reshape%d" % name)(out)
        x = tf.keras.layers.Multiply(name="scaling%d" % name)([x, out])
        x = layers.relu(name="enc_conv%d_2_relu" % name)(x + shortcut)
        return x

    def residual_block_first(self, x, ch, strides, name):
        if x.shape[-1] == ch:
            shortcut = layers.maxpool(k=strides, s=strides, name="max_pool%d" % name)
            shortcut = shortcut(x)
        else:
            shortcut = layers.conv(f=ch, k=1, s=strides, p="same", name="shortcut%d" % name)
            shortcut = shortcut(x)

        x = self.conv_bn_act(x=x, k=3, f=ch, s=strides, p="same", act=True, n="enc_conv%d_1" % name)
        x = self.conv_bn_act(x=x, k=3, f=ch, s=1, p="same", act=False, n="enc_conv%d_2" % name)
        act = layers.relu(name="enc_conv%d_2_relu" % name)
        x = act(x + shortcut)
        return x

    def residual_block_SE_first(self, x, ch, ratio, strides, name):
        if x.shape[-1] == ch:
            shortcut = layers.maxpool(k=strides, s=strides, name="max_pool_%d" % name)
            shortcut = shortcut(x)
        else:
            shortcut = layers.conv(f=ch, k=1, s=strides, p="same", name="shortcut_%d" % name)
            shortcut = shortcut(x)

        ch_reduced = ch // ratio

        x = self.conv_bn_act(x=x, k=3, f=ch, s=strides, p="same", act=True, n="enc_conv%d_1" % name)
        x = self.conv_bn_act(x=x, k=3, f=ch, s=1, p="same", act=False, n="enc_conv%d_2" % name)  # activation True or False

        squeeze = layers.global_avgpool(rank=3, name="GAP%d" % name)(x)
        excitation = layers.dense(f=ch_reduced, act="relu", b=False, name="excitation%d_1" % name)(squeeze)
        out = layers.dense(f=ch, act="sigmoid", b=False, name="excitation%d_2" % name)(excitation)

        out = tf.keras.layers.Reshape((1, 1, 1, ch), name="reshape%d" % name)(out)
        x = tf.keras.layers.Multiply(name="scaling%d" % name)([x, out])
        x = layers.relu(name="enc_conv%d_2_relu" % name)(x + shortcut)
        return x

    def build_model(self):
        if Mode == 0:
            enc_conv1_1 = self.conv_bn_act(x=self.input, f=ch, k=7, s=2, p="same", act=True, n="enc_conv1_1")
            max_pool1 = layers.maxpool(k=3, s=2, p="same", name="max_pool1")(enc_conv1_1)

            # conv2_x
            enc_conv2_block1 = self.residual_block(x=max_pool1, ch=ch, name=2)
            enc_conv2_block2 = self.residual_block(x=enc_conv2_block1, ch=ch, name=3)

            # conv3_x
            enc_conv3_block1 = self.residual_block_first(x=enc_conv2_block2, ch=ch*2, strides=2, name=4)
            enc_conv3_block2 = self.residual_block(x=enc_conv3_block1, ch=ch*2, name=5)

            # conv4_x
            enc_conv4_block1 = self.residual_block_first(x=enc_conv3_block2, ch=ch*4, strides=2, name=6)
            enc_conv4_block2 = self.residual_block(x=enc_conv4_block1, ch=ch*4, name=7)

            # conv5_x
            enc_conv5_block1 = self.residual_block_first(x=enc_conv4_block2, ch=ch*8, strides=2, name=8)
            enc_conv5_block2 = self.residual_block(x=enc_conv5_block1, ch=ch*8, name=9)

            gap = layers.global_avgpool(rank=3, name="gap")(enc_conv5_block2)
            dense = self.dense_layer(x=gap, f=2, act=False, n="dense_2")
            cls_out = layers.softmax(x=dense, name="softmax")

            self.cls_model = keras.Model({"cls_in": self.input}, {"cls_out": cls_out}, name="cls_model")

        elif Mode == 1:
            enc_conv1_1 = self.conv_bn_act(x=self.input, f=ch, k=7, s=2, p="same", act=True, n="enc_conv1_1")
            max_pool1 = layers.maxpool(k=3, s=2, p="same", name="max_pool1")(enc_conv1_1)

            # conv2_x
            enc_conv2_block1 = self.residual_SE_block(x=max_pool1, ch=ch, ratio=4, name=2)
            enc_conv2_block2 = self.residual_SE_block(x=enc_conv2_block1, ch=ch, ratio=4, name=3)

            # conv3_x
            enc_conv3_block1 = self.residual_block_SE_first(x=enc_conv2_block2, ch=ch * 2, ratio=4, strides=2, name=4)
            enc_conv3_block2 = self.residual_SE_block(x=enc_conv3_block1, ch=ch * 2, ratio=4, name=5)

            # conv4_x
            enc_conv4_block1 = self.residual_block_SE_first(x=enc_conv3_block2, ch=ch * 4, ratio=4, strides=2, name=6)
            enc_conv4_block2 = self.residual_SE_block(x=enc_conv4_block1, ch=ch * 4, ratio=4, name=7)

            # conv5_x
            enc_conv5_block1 = self.residual_block_SE_first(x=enc_conv4_block2, ch=ch * 8, ratio=4, strides=2, name=8)
            enc_conv5_block2 = self.residual_SE_block(x=enc_conv5_block1, ch=ch * 8, ratio=4, name=9)

            gap = layers.global_avgpool(rank=3, name="gap")(enc_conv5_block2)
            dense = self.dense_layer(x=gap, f=2, act=False, n="dense_2")
            cls_out = layers.softmax(x=dense, name="softmax")

            self.cls_model = keras.Model({"cls_in": self.input}, {"cls_out": cls_out}, name="cls_model")

        return self.cls_model

class Trainer:
    def __init__(self):
        self.lr = learning_rate
        self.file_name = file_name[file]
        self.decay = learning_decay
        self.epoch = epoch
        self.fold = fold
        self.batch_size = batch_size

        self.valid_acc, self.compare_acc, self.valid_loss, self.count = 0, 0, 0, 0
        tf.keras.backend.set_image_data_format("channels_last")
        self.valid_save, self.nii_save, self.model_select = False, False, False
        self.build_model()

        # self.path = os.path.join("/DataCommon/ksoh/AD_NC/ResNet18/" + self.file_name)
        self.path = os.path.join('/Users/오용석/Desktop/data_performance/' + self.file_name) # TODO: 데이터 저장할 path도 수정해주고
        if not os.path.exists(self.path): os.makedirs(self.path)

    def build_model(self):
        resnet = ResNet18()
        self.train_vars = []
        self.cls_model = resnet.build_model()
        self.train_vars += self.cls_model.trainable_variables

    def _train_one_batch(self, dat_all, lbl, gen_optim, train_vars, step, cv):
        with tf.GradientTape() as tape:
            res = self.cls_model({"cls_in": dat_all}, training=True)["cls_out"]
            loss = K.mean(keras.losses.binary_crossentropy(lbl, res))

        grads = tape.gradient(loss, train_vars)
        gen_optim.apply_gradients(zip(grads, train_vars))

        if step % 10 == 0:
            print("%dth iteration's training loss: %f", (step, loss))
            with self.train_summary_writer.as_default():
                tf.summary.scalar("%dfold_train_loss" % (cv + 1), loss, step=step)

    def _valid_logger(self, dat_all, lbl, epoch, cv):
        res = self.cls_model({"cls_in": dat_all}, training=False)["cls_out"]

        valid_loss = K.mean(keras.losses.binary_crossentropy(lbl, res))
        valid_acc = K.mean(K.equal(K.argmax(lbl, axis=-1), K.argmax(res, axis=-1)))

        self.valid_loss += valid_loss
        self.valid_acc += valid_acc
        self.count += 1

        print("%dth epoch's validation loss: %f, acc: %f", (epoch, self.valid_loss, self.valid_acc))

        if self.valid_save == True:
            self.valid_acc = self.valid_acc / self.count
            self.valid_loss = self.valid_loss / self.count

            if self.compare_acc <= self.valid_acc:
                self.model_select = True
                self.compare_acc = self.valid_acc

            elif self.valid_acc >= 0.70:
                self.model_select = True

            with self.valid_summary_writer.as_default():
                tf.summary.scalar("%dfold_valid_loss" % (cv + 1), self.valid_loss, step=epoch)
                tf.summary.scalar("%dfold_valid_acc" % (cv + 1), self.valid_acc, step=epoch)
                self.valid_acc, self.valid_loss, self.count = 0, 0, 0
                self.valid_save = False

    def train(self):
        util = Utils()
        dat, lbl = util.load_adni_data()

        for cv in range(0, 1): #TODO: 형은 5fold cross-validation 한번에 보려고 for문 돌린거
            self.train_summary_writer = tf.summary.create_file_writer(self.path + "/%dfold_train" % (cv + 1))
            self.valid_summary_writer = tf.summary.create_file_writer(self.path + "/%dfold_valid" % (cv + 1))
            self.test_summary_writer = tf.summary.create_file_writer(self.path + "/%dfold_test" % (cv + 1))
            self.train_all_idx, self.valid_all_idx, self.test_all_idx = util.data_permutation(lbl, cv)
            self.build_model()

            lr_schedule = keras.optimizers.schedules.ExponentialDecay(self.lr, decay_steps=len(self.train_all_idx) // self.batch_size, decay_rate=self.decay, staircase=True)
            optim = keras.optimizers.Adam(lr_schedule)
            global_step, self.compare_acc = 0, 0

            for cur_epoch in tqdm.trange(self.epoch, desc="ad_nc_resnet18_%s.py" % file_name[file]):
                self.train_all_idx = np.random.permutation(self.train_all_idx)

                # training -> 여기서 학습하고
                for cur_step in tqdm.trange(0, len(self.train_all_idx), self.batch_size, desc="%dfold_%depoch_%s" % (cv + 1, cur_epoch, self.file_name)):
                    cur_idx = self.train_all_idx[cur_step:cur_step + self.batch_size]
                    cur_dat, cur_lbl = util.seperate_data(cur_idx, dat, lbl, CENTER=False)

                    self._train_one_batch(dat_all=cur_dat, lbl=cur_lbl, gen_optim=optim, train_vars=self.train_vars, step=global_step, cv=cv)
                    global_step += 1

                # validation -> 한번의 iteration이 끝나면 그때의 모델 검증하고
                for val_step in tqdm.trange(0, len(self.valid_all_idx), self.batch_size, desc="Validation step: %dfold" % (cv + 1)):
                    val_idx = self.valid_all_idx[val_step:val_step + self.batch_size]
                    val_dat, val_lbl = util.seperate_data(val_idx, dat, lbl, CENTER=True)

                    if val_step + self.batch_size >= len(self.valid_all_idx): self.valid_save = True
                    self._valid_logger(dat_all=val_dat, lbl=val_lbl, epoch=cur_epoch, cv=cv)

                if self.model_select == True: # validation 성능이 이전보다 더 좋아지면 그 모델은 무조건 저장
                    self.cls_model.save(os.path.join(self.path + '/%dfold_cls_model_%03d' % (cv + 1, cur_epoch)))
                    self.model_select = False

                # Test -> 그 모델의 테스트 성능은 따로 보기 귀찮아서 매번 찍히도록 코딩했어
                tot_true, tot_pred = 0, 0
                for tst_step in tqdm.trange(0, len(self.test_all_idx), self.batch_size, desc="Testing step: %dfold" % (cv + 1)):
                    tst_idx = self.test_all_idx[tst_step:tst_step + self.batch_size]
                    tst_dat, tst_lbl = util.seperate_data(tst_idx, dat, lbl, CENTER=True)
                    res = self.cls_model({"cls_in": tst_dat}, training=False)["cls_out"]

                    if tst_step == 0:
                        tot_true, tot_pred = np.argmax(tst_lbl, axis=-1), np.argmax(res, axis=-1)
                    else:
                        tot_true = np.append(tot_true, np.argmax(tst_lbl, axis=-1))
                        tot_pred = np.append(tot_pred, np.argmax(res, axis=-1))

                acc, auc, sen, spe = self.evaluation_matrics(tot_true, tot_pred) # accuracy(acc), under roc curve (auc), sensitivity (sen), specificity (spe)
                print("%dth epoch's test acc: %f, auc: %f, sen: %f, spe: %f", (cur_epoch, acc, auc, sen, spe))
                with self.test_summary_writer.as_default():
                    tf.summary.scalar("%dfold_test_ACC" % (cv + 1), acc, step=cur_epoch)
                    tf.summary.scalar("%dfold_test_AUC" % (cv + 1), auc, step=cur_epoch)
                    tf.summary.scalar("%dfold_test_SEN" % (cv + 1), sen, step=cur_epoch)
                    tf.summary.scalar("%dfold_test_SPE" % (cv + 1), spe, step=cur_epoch)

    def evaluation_matrics(self, y_true, y_pred): # 평가 메트릭스
        acc = K.mean(K.equal(y_true, y_pred))
        auc = metrics.roc_auc_score(y_true, y_pred)
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        sen = tp / (tp + fn)
        spe = tn / (fp + tn)
        return acc, auc, sen, spe

    def MAUC(self, data, num_classes):
        """
        Calculates the MAUC over a set of multi-class probabilities and
        their labels. This is equation 7 in Hand and Till's 2001 paper.
        NB: The class labels should be in the set [0,n-1] where n = # of classes.
        The class probability should be at the index of its label in the
        probability list.
        I.e. With 3 classes the labels should be 0, 1, 2. The class probability
        for class '1' will be found in index 1 in the class probability list
        wrapped inside the zipped list with the labels.
        Args:
            data (list): A zipped list (NOT A GENERATOR) of the labels and the
                class probabilities in the form (m = # data instances):
                 [(label1, [p(x1c1), p(x1c2), ... p(x1cn)]),
                  (label2, [p(x2c1), p(x2c2), ... p(x2cn)])
                                 ...
                  (labelm, [p(xmc1), p(xmc2), ... (pxmcn)])
                 ]
            num_classes (int): The number of classes in the dataset.
        Returns:
            The MAUC as a floating point value.
        """
        # Find all pairwise comparisons of labels
        import itertools as itertools
        class_pairs = [x for x in itertools.combinations(range(num_classes), 2)]
        # Have to take average of A value with both classes acting as label 0 as this
        # gives different outputs for more than 2 classes
        sum_avals = 0
        for pairing in class_pairs:
            sum_avals += (self.a_value(data, zero_label=pairing[0], one_label=pairing[1]) + self.a_value(data, zero_label=pairing[1], one_label=pairing[0])) / 2.0
        return sum_avals * (2 / float(num_classes * (num_classes - 1)))  # Eqn 7 # 평가 메트릭스

    def a_value(self, probabilities, zero_label=0, one_label=1):
        """
        Approximates the AUC by the method described in Hand and Till 2001,
        equation 3.
        NB: The class labels should be in the set [0,n-1] where n = # of classes.
        The class probability should be at the index of its label in the
        probability list.
        I.e. With 3 classes the labels should be 0, 1, 2. The class probability
        for class '1' will be found in index 1 in the class probability list
        wrapped inside the zipped list with the labels.
        Args:
            probabilities (list): A zipped list of the labels and the
                class probabilities in the form (m = # data instances):
                 [(label1, [p(x1c1), p(x1c2), ... p(x1cn)]),
                  (label2, [p(x2c1), p(x2c2), ... p(x2cn)])
                                 ...
                  (labelm, [p(xmc1), p(xmc2), ... (pxmcn)])
                 ]
            zero_label (optional, int): The label to use as the class '0'.
                Must be an integer, see above for details.
            one_label (optional, int): The label to use as the class '1'.
                Must be an integer, see above for details.
        Returns:
            The A-value as a floating point.
        """
        # Obtain a list of the probabilities for the specified zero label class
        expanded_points = []
        for instance in probabilities:
            if instance[0] == zero_label or instance[0] == one_label:
                expanded_points.append((instance[0].item(), instance[zero_label + 1].item()))
        sorted_ranks = sorted(expanded_points, key=lambda x: x[1])
        n0, n1, sum_ranks = 0, 0, 0
        # Iterate through ranks and increment counters for overall count and ranks of class 0
        for index, point in enumerate(sorted_ranks):
            if point[0] == zero_label:
                n0 += 1
                sum_ranks += index + 1  # Add 1 as ranks are one-based
            elif point[0] == one_label:
                n1 += 1
            else:
                pass  # Not interested in this class
        # print('Before: n0', n0, 'n1', n1, 'n0*n1', n0*n1)
        if n0 == 0:
            n0 = 1e-10
        elif n1 == 0:
            n1 = 1e-10
        else:
            pass
        # print('After: n0', n0, 'n1', n1, 'n0*n1', n0*n1)
        return (sum_ranks - (n0 * (n0 + 1) / 2.0)) / float(n0 * n1)  # Eqn 3 # 평가 메트릭스

Tr = Trainer()
Tr.train()