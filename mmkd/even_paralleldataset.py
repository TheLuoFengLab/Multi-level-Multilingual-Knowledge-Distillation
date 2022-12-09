from torch.utils.data import DataLoader, Dataset
import random

# parallel corpus Dataset
class ParallelcorpusDataset(Dataset):
        def __init__(self, folder):
                super().__init__()
                self.folder = folder
                
                with open(folder+'europarl-v7.fr-en.en', 'r') as s_1:
                        # remove "\n" when readlines()
                        fr_en_en = s_1.read().split("\n")[:-1]
                with open(folder+'Europarl.bg-en.en', 'r') as s_2:
                        bg_en_en = s_2.read().split("\n")[:-1]
                with open(folder+'Europarl.de-en.en', 'r') as s_3:
                        de_en_en = s_3.read().split("\n")[:-1]
                with open(folder+'Europarl.en-es.en', 'r') as s_4:
                        en_es_en = s_4.read().split("\n")[:-1]
                with open(folder+'EUbookshop.bg-en.en', 'r') as s_5:
                        bg_en_en += s_5.read().split("\n")[:-1]
                with open(folder+'MultiUN.ar-en.en', 'r') as s_6:
                        ar_en_en = s_6.read().split("\n")[:2000000]
                with open(folder+'MultiUN.en-zh.en', 'r') as s_7:
                        en_zh_en = s_7.read().split("\n")[:2000000]
                with open(folder+'IITB.en-hi.en', 'r') as s_8:
                        en_hi_en = s_8.read().split("\n")[:-1]
                
                with open(folder+'europarl-v7.fr-en.fr', 'r') as t_1:
                        fr_en_fr = t_1.read().split("\n")[:-1]
                with open(folder+'Europarl.bg-en.bg', 'r') as t_2:
                        bg_en_bg = t_2.read().split("\n")[:-1]
                with open(folder+'Europarl.de-en.de', 'r') as t_3:
                        de_en_de = t_3.read().split("\n")[:-1]
                with open(folder+'Europarl.en-es.es', 'r') as t_4:
                        en_es_es = t_4.read().split("\n")[:-1]
                with open(folder+'EUbookshop.bg-en.bg', 'r') as t_5:
                        bg_en_bg += t_5.read().split("\n")[:-1]
                with open(folder+'MultiUN.ar-en.ar', 'r') as t_6:
                        ar_en_ar = t_6.read().split("\n")[:2000000]
                with open(folder+'MultiUN.en-zh.zh', 'r') as t_7:
                        en_zh_zh = t_7.read().split("\n")[:2000000]
                with open(folder+'IITB.en-hi.hi', 'r') as t_8:
                        en_hi_hi = t_8.read().split("\n")[:-1]
                
                source = [fr_en_en, bg_en_en, de_en_en, en_es_en, ar_en_en, en_zh_en, en_hi_en]
                target = [fr_en_fr, bg_en_bg, de_en_de, en_es_es, ar_en_ar, en_zh_zh, en_hi_hi]
                for i in range(len(source)):
                        # print(so,tar)
                        temp_so_tar = list(zip(source[i], target[i]))
                        random.shuffle(temp_so_tar)
                        temp_so, temp_tar = zip(*temp_so_tar)
                        source[i], target[i] = list(temp_so), list(temp_tar)
                self.source_data = []
                self.target_data = []
                for j in range(2000000):
                        for k in range(len(source)):
                                try:
                                        self.source_data.append(source[k][j])
                                        self.target_data.append(target[k][j])
                                except:
                                        pass
                        try:
                                self.source_data.append(source[j%7][j])
                                self.target_data.append(source[j%7][j])
                        except:
                                pass
                print(len(self.source_data))
        
        def __len__(self):
                return len(self.source_data)

        def __getitem__(self,index):
                source_sent = self.source_data[index]
                target_sent = self.target_data[index]
                return (source_sent,target_sent)
