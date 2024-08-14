from transformers import BertTokenizer, BertModel
from sklearn.cluster import DBSCAN
import torch, math, pandas

class Fmp_Bert:
    def __init__(self, model_name:str, file_path:str):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        with open(file_path, encoding='utf-8') as f:
            self.sentences = f.readlines()
        self.embeddings = {}
        self.dbscan = {}
        self.prob = {}
        self.entropy = {}
        self.model.eval()

    def embed(self):
        for sentence in self.sentences:
            encoded = self.tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
            subwords = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
            with torch.no_grad():
                output = self.model(**encoded)
            embedding = output.last_hidden_state.squeeze(0)
            for sw, emb in zip(subwords, embedding):
                emb = emb.detach().numpy()
                if sw not in self.embeddings:
                    self.embeddings[sw] = [emb]
                else:
                    self.embeddings[sw].append(emb)
    
    def cluster(self, min, brake):
        for sw in self.embeddings.keys():
            if len(self.embeddings[sw]) >= min:
                df_embedding = pandas.DataFrame(self.embeddings[sw]).T
                best_num_clusters = -1
                num_clusters = -1
                e = 0.5
                cnt = 0
                e_dbscan = None

                while num_clusters >= best_num_clusters and cnt < brake:
                    if num_clusters == best_num_clusters:
                        cnt += 1
                    else:
                        cnt = 0
                    best_dbscan = e_dbscan
                    e_dbscan = DBSCAN(eps=e, min_samples=2, metric='euclidean').fit_predict(df_embedding)
                    best_num_clusters = num_clusters
                    num_clusters = max(e_dbscan)
                    e += 0.5
                self.dbscan[sw] = list(best_dbscan)
                
                if list(best_dbscan).count(-1) != len(best_dbscan):
                    filtered_best_dbscan = [i for i in best_dbscan if i != -1]
                    list_prob = [filtered_best_dbscan.count(i)/len(filtered_best_dbscan) for i in range(max(filtered_best_dbscan)+1)]
                    self.prob[sw] = list_prob
                    self.entropy[sw] = sum([-p * math.log(p, 2) for p in list_prob])
