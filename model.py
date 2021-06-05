from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import pickle
from torchvision import transforms
from sklearn import metrics
import base64
from io import BytesIO


class Model:
    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(select_largest=False, device=device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        self.tr_raw = transforms.Compose([transforms.Resize(256), ])

        with open('data/db', 'rb') as f:
            self.data = pickle.load(f)

    def get_best_match(self, k, flag):
        sorted_embeddings = dict(enumerate(self.data[2][k]))
        sorted_embeddings = filter(lambda euc: euc != 0.0, sorted(sorted_embeddings.items(), key=lambda item: item[1]))
        best_id = [part[0] for part in sorted_embeddings]
        best_id = best_id[1+flag:6+flag]
        return best_id

    def add_photo(self, photo, name):
        flag = False
        k = 0
        self.photo = self.tr_raw(Image.open(photo))
        buffered = BytesIO()
        self.photo.save(buffered, format="JPEG")
        self.photo64 = base64.b64encode(buffered.getvalue())  # Первый элемент или photos64
        x = self.mtcnn(self.photo)
        with torch.no_grad():
            embeddings = self.resnet(x.unsqueeze(0))  # Второй элемент или embeddings
        for i in range(len(self.data[1])):
            if torch.all(torch.isclose(embeddings[0], self.data[1][i])):
                flag = True
                k = i
        if flag == False:
            self.data[0].append(self.photo64)
            self.data[1] = torch.cat([self.data[1], embeddings])  # Соединение embeddings для хранения
            self.data[2] = metrics.pairwise.euclidean_distances(self.data[1].numpy())  # Третий элемент или обновление euc
            self.data[3].append(name)
            k = -1

            with open('data/db', 'wb') as f:
                pickle.dump(self.data, f)

        best_id = self.get_best_match(k, flag)
        best_photos = []
        for i in range(len(best_id)):
            best_photos.append(self.data[0][best_id[i]])
        for i in range(len(best_photos)):
            best_photos[i] = best_photos[i].decode('utf-8')

        return best_photos

